"""Structured direct route: when it applies, what it costs, and its speed-coupled form.

The applicability tests and memory model of the per-``(species, x)``
block-Thomas route live here, beside the speed-coupled elimination below; the
factorizations and solves of the per-``(species, x)`` route stay in
:mod:`dkx.solve` until they are moved too (plan section 11.2).

**Speed-coupled elimination.**
The per-``(species, x)`` structured route (:func:`dkx.solve.build_tier1_solver`)
needs every collision term diagonal in ``(species, x)``, which pitch-angle
scattering is and the linearized Fokker-Planck and improved Sugama operators
are not: they are dense in ``(species, x)`` at each Legendre index ``l``, and
diagonal in ``l`` and in the angles.  With DKES trajectories and no tangential
drifts every other term is block-tridiagonal in ``l`` and diagonal in
``(species, x)``, so the whole f-block is block-tridiagonal in ``l`` with
blocks of size ``n_l = (active species-speed pairs at l) * Ntheta * Nzeta``.

A sweep over speed with the upper speed triangle moved to the right-hand side
would be cheaper, but it is not exact: in DKX's spectral speed basis the strict
lower triangle of the self-species matrix is 1-3% of its diagonal at ``l = 0``
and the electron-ion field-particle block has a lower triangle as large as its
diagonal (2026-10-05 probe), which is why the ``coarse_triangle`` Krylov
preconditioner still takes tens to thousands of iterations.  This module
therefore factors the coupled blocks exactly, one dense LU of size ``n_l`` per
``l``, with the ``Nxi_for_x`` ramp shrinking ``n_l`` where speeds drop out.

The constraint border ``[[A, B], [C, 0]]`` (any ``constraintScheme``) is
eliminated with ``A~ = A + gamma B C``, which lifts the density and energy null
space of a Fokker-Planck ``A``::

    t = A~^-1 (r + gamma B g),  (C A~^-1 B) s = C t - g,  f = t - A~^-1 B s.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.scipy.linalg import lu_factor, lu_solve

from dkx.drift_kinetic import KineticOperator


def coupled_available(op: KineticOperator) -> tuple[bool, str]:
    """Whether the speed-coupled structured direct route solves ``op`` exactly."""
    if op.fp is None and op.sugama is None:
        return False, "no dense collision operator (the per-(species, x) route applies)"
    if op.fp_phi1 is not None or op.include_phi1 or op.external_phi1_hat is not None:
        return False, "Phi1 couples the angles of every Legendre block"
    if op.point_at_x0:
        return False, "point_at_x0 x-grids give the x=0 rows a different form"
    try:
        _stripped(op)._check_block_extraction_supported()
    except NotImplementedError as exc:
        return False, str(exc)
    return True, ""


def _stripped(op: KineticOperator) -> KineticOperator:
    return replace(op, fp=None, sugama=None)


def _layout(op: KineticOperator) -> list[np.ndarray]:
    """Active flat ``(species, x)`` pair indices at every ``l`` that has any (static)."""
    n_s, n_x = op.n_species, op.n_x
    nxi = np.asarray(op.n_xi_for_x)
    return [
        np.array([s * n_x + x for s in range(n_s) for x in range(n_x) if nxi[x] > l], dtype=int)
        for l in range(min(op.n_xi, int(nxi.max())))
    ]


def coupled_peak_memory_bytes(op: KineticOperator) -> float:
    """Peak bytes of the speed-coupled factorization.

    The stored LU factors and off-diagonal blocks, times 2.8: the measured peak
    RSS above the JAX floor over that storage was 2.7-2.8 on three decks
    (0.22, 0.74 and 5.3 GB of storage; 2026-10-05), the dense elimination
    temporaries of the compiled program.
    """
    tz = op.n_theta * op.n_zeta
    pairs = np.array([len(a) for a in _layout(op)], dtype=float)
    stored = np.sum((pairs * tz) ** 2) + 2.0 * np.sum(pairs) * tz * tz
    return 8.0 * (2.8 * stored + 4.0 * op.f_size)


def coupled_flops(op: KineticOperator) -> float:
    """Elimination flop count: one LU and one ``n_l x n_(l+1)`` solve per ``l``."""
    tz = op.n_theta * op.n_zeta
    n = [len(a) * tz for a in _layout(op)]
    return float(sum((2.0 / 3.0) * a**3 + 2.0 * a * a * b for a, b in zip(n, n[1:] + [0])))


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class CoupledSolver:
    """Block LU over ``l`` of the speed-coupled f-block, plus the border."""

    op: KineticOperator
    lu: tuple  # per l: (lu, piv) of the Schur complement S_l
    lower: tuple  # per l: (k_l, TZ, TZ) blocks coupling row l to l - 1
    upper: tuple  # per l: (k_(l+1), TZ, TZ) blocks coupling row l to l + 1
    bc: tuple  # (B, C, gamma, Z, M_lu, Zt, Mt_lu)
    mom: tuple = ()  # (Q, gamma1, Zq, K_lu, Zqt, Kt_lu): the l = 1 momentum lift

    def tree_flatten(self):
        return (self.op, self.lu, self.lower, self.upper, self.bc, self.mom), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    # -- f-block with the gamma B C lift --------------------------------------
    def _a_solve(self, r: jnp.ndarray, transpose: bool) -> jnp.ndarray:
        """``A~^-1 r``: the lifted elimination, with its momentum lift undone.

        Woodbury, ``(M - g Q Q^T)^-1 = M^-1 + Z (I/g - Q^T Z)^-1 Z'^T``.
        """
        y = self._lifted_solve(r, transpose)
        if not self.mom:
            return y
        q, _g, zq, k_lu, zqt, kt_lu = self.mom
        z, kk = (zqt, kt_lu) if transpose else (zq, k_lu)
        return y + z @ lu_solve(kk, q.T @ y)

    def _lifted_solve(self, r: jnp.ndarray, transpose: bool) -> jnp.ndarray:
        op = self.op
        n_s, n_x, n_xi, n_t, n_z = op.f_shape
        tz = n_t * n_z
        layout = _layout(op)
        g = r.reshape(n_s * n_x, n_xi, tz)
        n_l = len(layout)
        rows = [g[layout[l], l].reshape(-1) for l in range(n_l)]
        sel = [np.searchsorted(layout[l - 1], layout[l]) for l in range(1, n_l)]

        def lo(l: int, y: jnp.ndarray) -> jnp.ndarray:  # Lo_l y_(l-1)
            yk = y.reshape(-1, tz)[sel[l - 1]]
            return jnp.einsum("kab,kb->ka", self.lower[l], yk).reshape(-1)

        def lo_t(l: int, x: jnp.ndarray) -> jnp.ndarray:  # Lo_l^T x_l -> l - 1
            out = jnp.zeros((len(layout[l - 1]), tz), dtype=x.dtype)
            return out.at[sel[l - 1]].set(
                jnp.einsum("kab,ka->kb", self.lower[l], x.reshape(-1, tz))
            ).reshape(-1)

        def up(l: int, x: jnp.ndarray) -> jnp.ndarray:  # U_l x_(l+1)
            out = jnp.zeros((len(layout[l]), tz), dtype=x.dtype)
            return out.at[sel[l]].set(
                jnp.einsum("kab,kb->ka", self.upper[l], x.reshape(-1, tz))
            ).reshape(-1)

        def up_t(l: int, y: jnp.ndarray) -> jnp.ndarray:  # U_l^T y_l -> l + 1
            yk = y.reshape(-1, tz)[sel[l]]
            return jnp.einsum("kab,ka->kb", self.upper[l], yk).reshape(-1)

        # A = U~ L~: U~ unit upper with U_l S_(l+1)^-1 above the diagonal, L~
        # lower with S_l on it and Lo_l below; elimination runs from l = Nxi - 1
        # down, so the singular l = 0 collision block is reached last.
        sol: list = [None] * n_l
        if not transpose:
            y = rows[:]
            for l in range(n_l - 2, -1, -1):
                y[l] = rows[l] - up(l, lu_solve(self.lu[l + 1], y[l + 1]))
            sol[0] = lu_solve(self.lu[0], y[0])
            for l in range(1, n_l):
                sol[l] = lu_solve(self.lu[l], y[l] - lo(l, sol[l - 1]))
        else:
            w = rows[:]
            w[-1] = lu_solve(self.lu[-1], rows[-1], trans=1)
            for l in range(n_l - 2, -1, -1):
                w[l] = lu_solve(self.lu[l], rows[l] - lo_t(l + 1, w[l + 1]), trans=1)
            sol[0] = w[0]
            for l in range(n_l - 1):
                sol[l + 1] = w[l + 1] - lu_solve(self.lu[l + 1], up_t(l, sol[l]), trans=1)
        # The truncated (x, l >= Nxi_for_x) DOFs carry identity rows: this is the
        # inverse of the pinned operator of dkx.solve._pinned_matvecs.
        out = g
        for l in range(n_l):
            out = out.at[layout[l], l].set(sol[l].reshape(-1, tz))
        return out.reshape(-1)

    def _solve1(self, r: jnp.ndarray, transpose: bool) -> jnp.ndarray:
        b, c, gamma, z, m_lu, zt, mt_lu = self.bc
        f_size = self.op.f_size
        if b.shape[1] == 0:
            return self._a_solve(r, transpose)
        rf, g = r[:f_size], r[f_size:]
        lift, test, zz, mm = (c.T, b.T, zt, mt_lu) if transpose else (b, c, z, m_lu)
        t = self._a_solve(rf + gamma * (lift @ g), transpose)
        s = lu_solve(mm, test @ t - g)
        return jnp.concatenate([t - zz @ s, s])

    def solve(self, r: jnp.ndarray, transpose: bool = False) -> jnp.ndarray:
        if r.ndim == 1:
            return self._solve1(r, transpose)
        return jax.vmap(lambda v: self._solve1(v, transpose), in_axes=1, out_axes=1)(r)


def _border(op: KineticOperator) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Dense ``B`` (f_size, E) and ``C`` (E, f_size) of the constraint border."""
    n_e = op.extra_size
    zero_f = jnp.zeros(op.f_shape, dtype=jnp.float64)

    def inject(extra):
        return op._source_and_constraint_rows(zero_f, zero_f, extra)[0].reshape(-1)

    def rows(f):
        return op._source_and_constraint_rows(zero_f, f.reshape(op.f_shape), jnp.zeros(n_e))[1]

    eye = jnp.eye(n_e, dtype=jnp.float64)
    b = jax.vmap(inject, out_axes=1)(eye) if n_e else jnp.zeros((op.f_size, 0))
    c = jax.jacrev(rows)(jnp.zeros(op.f_size)) if n_e else jnp.zeros((0, op.f_size))
    return b, c


def _factor(op: KineticOperator) -> CoupledSolver:
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    tz = n_t * n_z
    layout = _layout(op)
    blocks = _stripped(op).to_block_tridiagonal()  # (L, S, X, TZ, TZ)
    lower_all, diag_all, upper_all = (a.reshape(n_xi, n_s * n_x, tz, tz) for a in blocks)
    coll = op.fp if op.fp is not None else op.sugama
    cmat = jnp.transpose(coll.mat, (2, 0, 3, 1, 4)).reshape(n_xi, n_s * n_x, n_s * n_x)
    b, c = _border(op)
    b0 = b.reshape(n_s * n_x, n_xi, tz, -1)[:, 0].reshape(n_s * n_x * tz, -1)
    c0 = c.reshape(-1, n_s * n_x, n_xi, tz)[:, :, 0].reshape(-1, n_s * n_x * tz)
    eye_tz = jnp.eye(tz, dtype=jnp.float64)

    def diag_block(l: int) -> jnp.ndarray:
        act = layout[l]
        k = len(act)
        d = jnp.einsum("kab,kj->kajb", diag_all[l, act], jnp.eye(k))
        d = d + cmat[l][np.ix_(act, act)][:, None, :, None] * eye_tz[None, :, None, :]
        return d.reshape(k * tz, k * tz)

    d0 = diag_block(0)
    if b.shape[1]:
        bc0 = b0 @ c0
        gamma = jnp.mean(jnp.abs(jnp.diagonal(d0))) / jnp.max(jnp.abs(bc0))
        d0 = d0 + gamma * bc0
    else:
        gamma = jnp.asarray(1.0)
    # Momentum conservation leaves the l = 1 collision block singular at every
    # angle; the Schur complement from l >= 2 lifts all of that but a parallel
    # flow per species, which g Q Q^T lifts and _a_solve restores (Woodbury).
    q = None
    if len(layout) > 1:
        ang = (op._fs_average_factor() * op.b_hat).reshape(-1)
        prof = op.x**3 * op.x_weights
        q_all = jnp.einsum("st,x,a->sxat", jnp.eye(n_s), prof, ang).reshape(n_s * n_x, tz, n_s)
        q = q_all[layout[1]].reshape(-1, n_s)  # the speeds that carry l = 1
        qq = q @ q.T
    lowers = [jnp.zeros((0, tz, tz))] + [lower_all[l, layout[l]] for l in range(1, len(layout))]
    uppers = [upper_all[l, layout[l + 1]] for l in range(len(layout) - 1)] + [jnp.zeros((0, tz, tz))]
    lus: list = [None] * len(layout)
    lus[-1] = lu_factor(d0 if len(layout) == 1 else diag_block(len(layout) - 1))
    for l in range(len(layout) - 2, -1, -1):
        sel = np.searchsorted(layout[l], layout[l + 1])
        k_l, k_n = len(layout[l]), len(sel)
        # W = S_(l+1)^-1 Lo_(l+1), then S_l = D_l - U_l W on the rows U_l reaches.
        place = jnp.eye(k_l)[:, sel].T  # (k_(l+1), k_l)
        lo_dense = jnp.einsum("kab,kj->kajb", lowers[l + 1], place).reshape(k_n * tz, k_l * tz)
        w = lu_solve(lus[l + 1], lo_dense).reshape(k_n, tz, k_l * tz)
        uw = jnp.einsum("kab,kbn->kan", uppers[l], w)
        d = d0 if l == 0 else diag_block(l)
        if l == 1 and q is not None:
            gamma1 = jnp.mean(jnp.abs(jnp.diagonal(d))) / jnp.max(jnp.abs(qq))
            d = d + gamma1 * qq
        d = d.reshape(k_l, tz, k_l * tz)
        lus[l] = lu_factor(d.at[sel].add(-uw).reshape(k_l * tz, k_l * tz))
    solver = CoupledSolver(op, tuple(lus), tuple(lowers), tuple(uppers), (b, c, gamma) + (None,) * 4)
    if q is not None:
        qf = jnp.zeros((n_s * n_x, n_xi, tz, n_s)).at[layout[1], 1].set(q.reshape(-1, tz, n_s))
        qf = qf.reshape(-1, n_s)
        zq, zqt = (
            jax.vmap(lambda v, t=t: solver._lifted_solve(v, t), in_axes=1, out_axes=1)(qf)
            for t in (False, True)
        )
        eye = jnp.eye(n_s) / gamma1
        mom = (qf, gamma1, zq, lu_factor(eye - qf.T @ zq), zqt, lu_factor(eye - qf.T @ zqt))
        solver = replace(solver, mom=mom)
    if not b.shape[1]:
        return replace(solver, bc=(b, c, gamma, b, None, b, None))
    z = jax.vmap(lambda v: solver._a_solve(v, False), in_axes=1, out_axes=1)(b)
    zt = jax.vmap(lambda v: solver._a_solve(v, True), in_axes=1, out_axes=1)(c.T)
    bc = (b, c, gamma, z, lu_factor(c @ z), zt, lu_factor(b.T @ zt))
    return replace(solver, bc=bc)


_factor_compiled = jax.jit(_factor)


def build_coupled_solver(op: KineticOperator) -> CoupledSolver:
    """Factor the speed-coupled structured direct solver for ``op``."""
    ok, reason = coupled_available(op)
    if not ok:
        raise NotImplementedError(f"speed-coupled structured direct route unavailable: {reason}")
    return _factor_compiled(op)


# Default memory budget above which ``solve(method="auto")`` prefers the
# memory-lean truncated structured direct kernel over the full-band
# factorization.  Chosen to match the validated HSX head-to-head benchmark
# (tools/benchmarks/tier1_hsx_head_to_head.py).  Overridable per call via the
# ``tier1_memory_budget_gb`` argument or the environment variable below.
_TIER1_BUDGET_GB_DEFAULT = 8.0
_TIER1_BUDGET_ENV = "DKX_TIER1_MEMORY_BUDGET_GB"

# RHSMode 1/2/3 drives (radial gradient on L=0,2; inductive E_parallel on L=1)
# and every RHSMode 1/2/3 output moment (fluxes, flows, sources, FSA
# constraints) live on the lowest three Legendre modes, so keeping three
# solution blocks is exact for the standard transport quantities.
_TIER1_KEEP_LOWEST_DEFAULT = 3


def _is_traced(*arrays: Any) -> bool:
    return any(isinstance(a, jax.core.Tracer) for a in arrays)




# =============================================================================
# Structured direct (block Thomas over Legendre modes)
# =============================================================================


def tier1_available(op: KineticOperator) -> tuple[bool, str]:
    """Check whether the structured direct family applies to ``op``.

    The decision is driven by the operator's own block extraction: if
    :meth:`KineticOperator.legendre_blocks` refuses (Er L±2 terms,
    Fokker-Planck collisions), the structured direct route is off.  On top of
    that the bordered constraint machinery must be diagonal over (species, x)
    (``constraintScheme`` 0 or 2 without ``point_at_x0``).  Non-uniform
    ``Nxi_for_x`` (the production speed-dependent Legendre ramp) is accepted:
    every (species, x) subsystem is closed, so the truncated structured direct
    kernel solves it with its own ``n_blocks = Nxi_for_x[ix]`` — exactly the
    packed Fortran system.  Only the full-band factorization
    (:func:`build_tier1_solver`) additionally requires uniform ``Nxi_for_x``;
    ramped decks always route through the truncated kernel.
    """
    try:
        op._check_block_extraction_supported()
    except NotImplementedError as exc:
        return False, str(exc)
    if op.constraint_scheme not in (0, 2):
        return False, (
            f"constraintScheme={op.constraint_scheme} borders couple speed nodes; "
            "only 0 and 2 keep the (species, x) block split exact"
        )
    if op.constraint_scheme == 2 and op.point_at_x0:
        return False, "point_at_x0 x-grids give the x=0 constraint row a different form"
    return True, ""


# Elimination work per right-hand side above which the speed-coupled route
# yields to recycled Krylov. Measured on eight cores of the benchmark host
# (2026-10-05): the direct route wins at 0.1-60 GFlop (quick 2-species FP
# 0.18 s against 0.50 s, transportMatrix_geometryScheme2 2.8 s against 33 s at
# 1,181 GCROT iterations) and at 279 GFlop over two right-hand sides
# (geometryScheme11 9.9 s against 48 s), and loses at 560 GFlop with one
# (geometryScheme4_2species_noEr 19.8 s against 7.1 s at 26 iterations).
_COUPLED_GFLOP_PER_RHS = 150.0


def _coupled_route_fits(op: KineticOperator, budget_gb: float | None, n_rhs: int | None = None) -> bool:
    """Whether ``method="auto"`` takes the speed-coupled structured direct route."""
    if not coupled_available(op)[0]:
        return False
    if n_rhs is None:
        n_rhs = {2: 3, 3: 2}.get(int(op.rhs_mode), 1)
    budget_bytes, _ = _tier1_budget_bytes(budget_gb)
    return (
        coupled_peak_memory_bytes(op) <= budget_bytes
        and coupled_flops(op) <= _COUPLED_GFLOP_PER_RHS * 1e9 * max(1, n_rhs)
    )


def _uniform_nxi_for_x(op: KineticOperator) -> bool:
    """Whether every speed node retains the full Legendre resolution."""
    return int(np.min(np.asarray(op.n_xi_for_x))) >= op.n_xi


# =============================================================================
# Structured direct memory model and the full-vs-truncated route decision
# =============================================================================


def tier1_full_band_bytes(op: KineticOperator) -> float:
    """Bytes of the structured direct Legendre bands (``lower``/``diag``/``upper``).

    :func:`build_tier1_solver` materializes the three block-tridiagonal bands
    of :meth:`KineticOperator.to_block_tridiagonal`, each of shape
    ``(n_xi, n_species, n_x, m, m)`` with block dimension ``m = n_theta *
    n_zeta`` (the dense theta*zeta angular block per Legendre mode, per
    (species, x) subsystem), in float64::

        bytes = 3 * sum_x(Nxi_for_x) * n_species * (n_theta * n_zeta)**2 * 8

    The leading ``3`` counts ``lower``, ``diag`` and ``upper``; a subsystem at
    speed node ``ix`` carries only its own ``Nxi_for_x[ix]`` Legendre blocks
    (``sum_x(Nxi_for_x) = n_xi * n_x`` for uniform ``Nxi_for_x``).  This is
    the ~39 GB figure for the 744k-unknown uniform HSX case (n_theta=25,
    n_zeta=51, n_xi=100, n_x=5, n_species=2).
    """
    m = float(op.n_theta * op.n_zeta)
    n_blocks_total = float(np.sum(np.asarray(op.n_xi_for_x)))
    return 3.0 * n_blocks_total * float(op.n_species) * m * m * 8.0


def tier1_peak_memory_bytes(op: KineticOperator) -> float:
    """Peak-memory estimate of the full structured direct factorization.

    Adds the block-Thomas LU factors and elimination temporaries on top of the
    three input bands (:func:`tier1_full_band_bytes`).  The
    ``BlockTridiagFactors`` store the per-block LU factors plus the two
    off-diagonal bands (~2x the band storage), and the vmapped sweep holds a
    few block temporaries live, so the peak is estimated at ``2.5x`` the band
    storage — the multiplier used by the validated HSX benchmark.
    """
    return 2.5 * tier1_full_band_bytes(op)


def tier1_truncated_peak_memory_bytes(
    op: KineticOperator,
    keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
    subsystem_batch: int | str = "auto",
) -> float:
    """Working-set estimate of :func:`_solve_tier1_truncated` (structured direct).

    The truncated route never materializes the full Legendre bands, so the
    ~``tier1_peak_memory_bytes`` full-band peak wildly overestimates it (46x on
    the 1.27M-DOF production deck).  Its live buffers, with block dimension
    ``m = n_theta * n_zeta``, subsystem batch ``B = n_species * n_x``, and
    concurrent elimination width ``w = subsystem_batch`` (float64, 8 bytes
    each):

    * the compact coefficient set (:func:`_truncated_coefficients`): the two
      angular derivative matrices, the ExB matrix, the kron assembly
      temporaries, and the per-species streaming matrices —
      ``(5 + n_species) * m^2`` entries;
    * the per-subsystem broadcast of the streaming matrix
      (``jnp.repeat`` to the ``B`` axis) — ``B * m^2`` entries;
    * ``w`` concurrent ``solvax.direct.block_thomas_truncated_fn`` sweeps
      (the batched ``jax.lax.map(..., batch_size=w)`` elimination in
      :func:`_solve_tier1_truncated`): per subsystem the LU carry, the
      assembled ``(L, D, U)`` block triple, elimination temporaries, and the
      stacked ``keep`` head factors — ``w * (2 * keep + 8) * m^2`` entries,
      doubled for the ``jax.lax.map`` pipeline (one batch in flight while the
      next is staged; ``w = 1`` is the fully serial sweep);
    * the state buffers (zero-padded full-shape solution, its RHS reshape,
      and the assembly/concat copies) — ``4 * total_size`` entries.

    ``subsystem_batch="auto"`` models the width the solve itself resolves
    (:func:`_resolve_subsystem_batch`: width 1 on the CPU backend, the
    memory-budgeted :func:`tier1_truncated_subsystem_width` on accelerators);
    an integer models that fixed width (clamped to ``[1, B]``).  The sum is
    doubled as a safety margin for allocator slack and XLA fusion
    temporaries.  Validated against measured process peaks on the profiling
    deck ladder (production 1.27M / mid 337k / small 41k DOFs): the estimate
    lands within about 1.1-1.5x of measurement, on the high side.
    """
    m = float(op.n_theta * op.n_zeta)
    mm_bytes = m * m * 8.0
    n_s = float(op.n_species)
    batch = n_s * float(op.n_x)
    keep = float(min(int(keep_lowest), int(op.n_xi)))
    if isinstance(subsystem_batch, str):
        width = float(_resolve_subsystem_batch(op, subsystem_batch, int(keep)))
    else:
        width = float(max(1, min(int(subsystem_batch), int(batch))))
    coeff_bytes = (5.0 + n_s) * mm_bytes
    stream_broadcast_bytes = batch * mm_bytes
    sweep_bytes = 2.0 * width * (2.0 * keep + 8.0) * mm_bytes
    state_bytes = 4.0 * float(op.total_size) * 8.0
    return 2.0 * (coeff_bytes + stream_broadcast_bytes + sweep_bytes + state_bytes)


def tier1_truncated_subsystem_width(
    op: KineticOperator,
    keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
    memory_budget_gb: float | None = None,
) -> int:
    """Largest subsystem batch width whose modeled footprint fits the budget.

    The memory-aware chooser behind ``subsystem_batch="auto"`` on accelerator
    backends: the widest ``w in [1, B]`` (``B = n_species * n_x``) such that
    :func:`tier1_truncated_peak_memory_bytes` with ``subsystem_batch=w`` stays
    within :func:`dkx.batch.resolve_memory_budget_bytes` — an explicit
    ``memory_budget_gb``, else a fraction of the device/host memory.  Width 1
    reproduces the fully serial per-subsystem elimination, so a tight budget
    degrades gracefully to the minimum-memory behavior.
    """
    from dkx.batch import resolve_memory_budget_bytes  # local import: batch imports solve

    budget = resolve_memory_budget_bytes(memory_budget_gb)
    b = max(1, int(op.n_species) * int(op.n_x))
    for width in range(b, 1, -1):
        if (
            tier1_truncated_peak_memory_bytes(op, keep_lowest, subsystem_batch=width)
            <= budget
        ):
            return width
    return 1


def _resolve_subsystem_batch(
    op: KineticOperator, subsystem_batch: int | str, keep: int
) -> int:
    """Map the ``solve(subsystem_batch=...)`` knob to a concrete width.

    ``"auto"`` is backend-aware:

    * CPU backend — width 1, the fully serial sweep.  Measured on the
      10-core M4 profiling host (336,610-DOF hsx_pas_dkes_mid warm solves,
      8 threads): every width > 1 is neutral-to-slower than width 1 (ramped
      deck 10.3 s at width 1 vs 11.4 s grouped width 2; uniform-Nxi variant
      16.6 s at width 1 vs 20.5 s at width 10), because XLA:CPU executes the
      batch axis of the LAPACK factor/solve custom calls serially per
      element with extra cache pressure — the batched sweep adds memory,
      not CPU parallelism.
    * accelerator backends — the widest width whose modeled footprint fits
      the memory budget (:func:`tier1_truncated_subsystem_width`); batched
      scans raise device occupancy there, and the budget clamp bounds the
      working set.
    """
    if isinstance(subsystem_batch, str):
        if subsystem_batch.strip().lower() != "auto":
            raise ValueError(
                f"unknown subsystem_batch {subsystem_batch!r}; expected 'auto' "
                "or a positive integer width"
            )
        if jax.default_backend() == "cpu":
            return 1
        return tier1_truncated_subsystem_width(op, keep_lowest=keep)
    width = int(subsystem_batch)
    if width < 1:
        raise ValueError(f"subsystem_batch must be >= 1, got {width}")
    return min(width, max(1, int(op.n_species) * int(op.n_x)))


def _tier1_budget_bytes(budget_gb: float | None) -> tuple[float, float]:
    """Resolve the truncation budget (bytes, GB) from arg / env / default."""
    if budget_gb is None:
        env = os.environ.get(_TIER1_BUDGET_ENV)
        budget_gb = float(env) if env not in (None, "") else _TIER1_BUDGET_GB_DEFAULT
    return float(budget_gb) * 2.0**30, float(budget_gb)


def _truncation_supported(op: KineticOperator, keep: int) -> tuple[bool, str]:
    """Structural check that the truncated structured direct kernel applies.

    Assumes :func:`tier1_available` already passed (PAS/DKES family,
    constraintScheme in {0, 2}, no point_at_x0).  Additionally every closed
    (species, x) subsystem must retain at least ``keep`` Legendre blocks,
    unless full recovery (``keep == n_xi``) is requested. Full recovery retains
    each chain's own ``Nxi_for_x[ix]`` blocks and pads only inactive DOFs.
    """
    if op.constraint_scheme not in (0, 2):
        return (
            False,
            f"constraintScheme={op.constraint_scheme} border couples Legendre modes",
        )
    if op.point_at_x0:
        return False, "point_at_x0 x-grids are not handled by the truncated kernel"
    if keep > op.n_xi:
        return False, f"keep_lowest={keep} exceeds Nxi={op.n_xi}"
    if keep != op.n_xi and int(np.min(np.asarray(op.n_xi_for_x))) < keep:
        return (
            False,
            f"min Nxi_for_x={int(np.min(np.asarray(op.n_xi_for_x)))} < keep_lowest={keep}",
        )
    return True, ""


def _rhs_confined_to_lowest_blocks(
    op: KineticOperator, rhs2d: jnp.ndarray, keep: int
) -> bool | None:
    """Whether the RHS has Legendre support only on modes ``l < keep``.

    Returns ``None`` when ``rhs2d`` is a tracer (support cannot be read under
    jit/grad); callers then fall back to the structural ``rhs_mode`` guarantee.
    The truncated kernel computes exactly the lowest ``keep`` Legendre blocks
    and zero-pads the rest, so it is exact iff both the drive and the requested
    output moments live on ``l < keep`` — true for the RHSMode 1/2/3 transport
    drives and their fluxes/flows/sources, which touch only ``l <= 2``.
    """
    if _is_traced(rhs2d):
        return None
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    if keep >= n_xi:
        return True
    f = np.asarray(rhs2d)[: op.f_size].reshape(n_s, n_x, n_xi, n_t * n_z, -1)
    return bool(np.max(np.abs(f[:, :, keep:])) == 0.0)
