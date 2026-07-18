#!/usr/bin/env python
"""Head-to-head benchmark: dkx tier-1 truncated solve vs the Fortran baseline.

Case: HSX PAS DKES RHSMode=1 (Ntheta=25, Nzeta=51, Nxi=100, Nx=5, two species;
744,610 unknowns with the Fortran Nxi_for_x ramp).  The Fortran baseline
(measured on the same MacBook) is 463.6 s / 3.98 GB peak RSS on 1 MPI rank and
229.5 s / 2.86 GB on 2 ranks.

Route selection: for the PAS/DKES family the (species, x) axes are uncoupled
Legendre-block-tridiagonal systems with dense TZ = Ntheta*Nzeta blocks.  The
full tier-1 factor storage is ``3 * Nxi * S * X * TZ^2 * 8 B`` (~39 GB here),
so when that estimate breaches the memory budget this benchmark uses
``solvax.direct.block_thomas_truncated_fn``: blocks are assembled on the fly
from the analytic per-term coefficients of
``dkx.drift_kinetic.KineticOperator.legendre_blocks`` (never
materializing the bands), and only the lowest ``keep_lowest=3`` Legendre
blocks of the solution are computed.  That is exact for every RHSMode=1
output moment (fluxes, flows, sources, FSA constraints): the drives have
Legendre support l <= 2 and all moments contract against l <= 2 only.
When the full tier-1 estimate fits the budget, the canonical
``dkx.solve.solve(..., method="block_tridiagonal")`` path is used
instead.

The input is run with ``Nxi_for_x_option = 0`` (uniform Nxi_for_x) because the
canonical tier-1 elimination requires all speed nodes to retain the full
Legendre resolution.  The Fortran baseline uses the default ramp (option 1),
which *drops* high-L modes at low x — so the JAX problem solved here is
strictly LARGER than the Fortran one (bias against JAX; small physics
differences at the discretization level are expected).  ``--ramp`` keeps the
namelist's ramp and solves the exact Fortran discretization through the
truncated kernel (n_blocks = Nxi_for_x[ix] per subsystem).

Usage::

    /usr/bin/time -l micromamba run -n dkx python \
        tools/benchmarks/tier1_hsx_head_to_head.py \
        --input /Users/rogerio/local/fortran_scaling_baseline/sized/input.namelist \
        --fortran-h5 /Users/rogerio/local/fortran_scaling_baseline/sized/sfincsOutput.h5

    # cross-check of the truncated kernel against the full tier-1 factorization
    # on a reduced grid (memory-safe):
    ... tier1_hsx_head_to_head.py --input ... --validate-small
    # bound the Nxi_for_x ramp-vs-uniform physics effect on a reduced grid:
    ... tier1_hsx_head_to_head.py --input ... --ramp-effect
"""

from __future__ import annotations

import argparse
import os
import re
import resource
import sys
import time
from pathlib import Path

# --device must take effect before the jax backend initializes.
if "--device" in sys.argv:
    _dev = sys.argv[sys.argv.index("--device") + 1]
    if _dev not in ("cpu", "gpu"):
        raise SystemExit(f"--device must be cpu or gpu, got {_dev!r}")
    os.environ.setdefault("JAX_PLATFORMS", {"cpu": "cpu", "gpu": "cuda"}[_dev])

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solvax.direct import block_thomas_truncated_fn  # noqa: E402

from dkx.drift_kinetic import KineticOperator  # noqa: E402
from dkx.moments import (  # noqa: E402
    FluxSurface,
    SpeciesParams,
    StateLayout,
    VelocityGrid,
    rhsmode1_moments,
)
from dkx.namelist import parse_sfincs_input_text  # noqa: E402

KEEP_LOWEST = 3  # RHSMode=1 drives and output moments live on l <= 2


def rss_gb() -> float:
    """Current process peak RSS in GB (ru_maxrss is bytes on macOS, KB on Linux)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 2**30 if sys.platform == "darwin" else v / 2**20


def gpu_mem_gb() -> float | None:
    """Bytes in use on the first accelerator device, if any."""
    dev = jax.devices()[0]
    if dev.platform not in ("gpu", "cuda", "rocm"):
        return None
    stats = dev.memory_stats()
    if not stats:
        return None
    return stats.get("bytes_in_use", 0) / 2**30


def patch_namelist(text: str, group: str, settings: dict[str, str]) -> str:
    """Insert/override ``key = value`` lines in one namelist group."""
    out = text
    for key, value in settings.items():
        pat = re.compile(rf"^\s*{re.escape(key)}\s*=.*$", re.IGNORECASE | re.MULTILINE)
        if pat.search(out):
            out = pat.sub(f"  {key} = {value}", out)
        else:
            gpat = re.compile(rf"(^\s*&{re.escape(group)}\s*$)", re.IGNORECASE | re.MULTILINE)
            if not gpat.search(out):
                raise ValueError(f"group &{group} not found in namelist text")
            out = gpat.sub(rf"\1\n  {key} = {value}", out, count=1)
    return out


def full_band_gigabytes(op: KineticOperator) -> float:
    """Storage of the full tier-1 bands (lower/diag/upper), before factors."""
    n_tz = op.n_theta * op.n_zeta
    return 3.0 * op.n_xi * op.n_species * op.n_x * n_tz * n_tz * 8.0 / 2**30


# =============================================================================
# Truncated tier-1 solve: analytic on-the-fly blocks + block_thomas_truncated_fn
# =============================================================================


class TruncatedTier1:
    """Memory-lean tier-1 solve for the PAS/DKES family.

    Mirrors ``dkx.solve.build_tier1_solver`` (same analytic blocks as
    ``KineticOperator.legendre_blocks``, same exact rank-one border absorption
    ``A~ = A + gamma B C``) but assembles the (m, m) blocks per Legendre index
    inside ``solvax.direct.block_thomas_truncated_fn``, so peak memory is
    O(keep_lowest * m^2) per (species, x) subsystem instead of O(Nxi * m^2).
    Returns only Legendre blocks l < keep_lowest (exact); higher blocks are
    zero-padded in the assembled state vector.
    """

    def __init__(self, op: KineticOperator, keep_lowest: int = KEEP_LOWEST):
        op._check_block_extraction_supported()  # PAS/DKES family only
        if op.constraint_scheme != 2:
            raise NotImplementedError("this benchmark path expects constraintScheme=2")
        if op.point_at_x0:
            raise NotImplementedError("point_at_x0 x-grids are not handled here")
        # Non-uniform Nxi_for_x (the Fortran ramp) is supported: each (species, x)
        # subsystem is closed, so it is solved with n_blocks = Nxi_for_x[ix]
        # (modes above are excluded unknowns, exactly as in Fortran); needs
        # Nxi_for_x[ix] >= keep_lowest everywhere.
        self.n_blocks_per_x = [int(v) for v in np.asarray(op.n_xi_for_x)]
        if min(self.n_blocks_per_x) < int(keep_lowest):
            raise NotImplementedError(
                f"min Nxi_for_x = {min(self.n_blocks_per_x)} < keep_lowest = {keep_lowest}"
            )
        self.op = op
        self.keep = int(keep_lowest)
        n_tz = op.n_theta * op.n_zeta

        # ---- per-term coefficient matrices (exactly as legendre_blocks) ----
        eye_t = jnp.eye(op.n_theta, dtype=jnp.float64)
        eye_z = jnp.eye(op.n_zeta, dtype=jnp.float64)
        d_theta_tz = jnp.kron(op.ddtheta, eye_z)
        d_zeta_tz = jnp.kron(eye_t, op.ddzeta)

        sqrt_t_over_m = jnp.sqrt(op.t_hat / op.m_hat)  # (S,)
        v_theta = (op.b_hat_sup_theta / op.b_hat).reshape((-1,))
        v_zeta = (op.b_hat_sup_zeta / op.b_hat).reshape((-1,))
        self.stream_s = (
            sqrt_t_over_m[:, None, None]
            * (v_theta[None, :, None] * d_theta_tz[None, :, :] + v_zeta[None, :, None] * d_zeta_tz[None, :, :])
        )  # (S, TZ, TZ)
        mirror_geom = op.b_hat_sup_theta * op.db_hat_dtheta + op.b_hat_sup_zeta * op.db_hat_dzeta
        self.mirror_s = (
            -sqrt_t_over_m[:, None] * (mirror_geom / (2.0 * op.b_hat**2)).reshape((-1,))[None, :]
        )  # (S, TZ) diagonal
        if op.with_exb:
            coef_theta, coef_zeta = op._exb_coefficients()
            self.exb = (
                coef_theta.reshape((-1,))[:, None] * d_theta_tz
                + coef_zeta.reshape((-1,))[:, None] * d_zeta_tz
            )  # (TZ, TZ)
        else:
            self.exb = jnp.zeros((n_tz, n_tz), dtype=jnp.float64)
        self.pas_coef = op.pas.coef  # (S, X, L)
        self.cl = op.xi_coupling_lower  # (L,)
        self.cu = op.xi_coupling_upper  # (L,)

        # ---- border: source column b0 (l=0 rows), constraint row c0 ----
        self.b0 = jnp.ones((n_tz,), dtype=jnp.float64)
        self.c0 = op._fs_average_factor().reshape((-1,))
        # gamma per (S, X): conditioning-friendly scale of the rank-one update
        # (any nonzero value is algebraically exact).
        exb_diag_mean = jnp.mean(jnp.abs(jnp.diagonal(self.exb)))
        scale = jnp.mean(jnp.abs(self.pas_coef), axis=2) + exb_diag_mean  # (S,X)
        scale = jnp.where(scale > 0.0, scale, 1.0)
        self.gamma = scale / jnp.max(jnp.abs(self.c0))  # (S,X)

        self._solve_one = jax.jit(self._solve_one_impl, static_argnums=(6,))

    # -- one (species, x) subsystem ------------------------------------------
    def _blocks_at(
        self,
        k: jnp.ndarray,
        stream: jnp.ndarray,
        mirror: jnp.ndarray,
        pas_row: jnp.ndarray,
        x_val: jnp.ndarray,
        gamma: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Analytic (L_k, D_k, U_k) of Legendre row k (within one closed subsystem)."""
        n_tz = stream.shape[0]
        idx = jnp.arange(n_tz)
        kf = k.astype(jnp.float64)
        # lower: row k <- col k-1 (streaming + mirror), scaled by x
        cl_k = jnp.take(self.cl, k)
        lower = (x_val * cl_k) * stream
        lower = lower.at[idx, idx].add((x_val * (-cl_k * (kf - 1.0))) * mirror)
        # upper: row k <- col k+1
        cu_k = jnp.take(self.cu, jnp.minimum(k, self.op.n_xi - 1))
        upper = (x_val * cu_k) * stream
        upper = upper.at[idx, idx].add((x_val * (cu_k * (kf + 2.0))) * mirror)
        # diag: ExB + PAS
        diag = self.exb.at[idx, idx].add(jnp.take(pas_row, k))
        # rank-one border absorption on the l=0 diagonal block
        diag = jnp.where(k == 0, diag + gamma * jnp.outer(self.b0, self.c0), diag)
        return lower, diag, upper

    def _solve_one_impl(
        self,
        stream: jnp.ndarray,
        mirror: jnp.ndarray,
        pas_row: jnp.ndarray,
        x_val: jnp.ndarray,
        gamma: jnp.ndarray,
        rhs_low: jnp.ndarray,  # (keep, TZ, n_rhs) — includes the z-column (b0)
        n_blocks: int,
    ) -> jnp.ndarray:
        def block_fn(k: jnp.ndarray):
            return self._blocks_at(k, stream, mirror, pas_row, x_val, gamma)

        return block_thomas_truncated_fn(block_fn, n_blocks, rhs_low, self.keep)

    # -- full bordered solve ---------------------------------------------------
    def solve(self, rhs: jnp.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Solve K x = rhs; return the zero-padded full state vector (numpy).

        Legendre blocks l >= keep_lowest of f are ZERO (not solved); the
        border/source unknowns are exact.  Valid for l <= 2 output moments.
        """
        op = self.op
        n_s, n_x, n_xi, n_t, n_z = op.f_shape
        n_tz = n_t * n_z
        rhs = jnp.asarray(rhs, dtype=jnp.float64)
        rhs_f = rhs[: op.f_size].reshape(n_s, n_x, n_xi, n_tz)
        r_c = np.asarray(rhs[op.f_size :]).reshape(n_s, n_x)
        tail = float(jnp.max(jnp.abs(rhs_f[:, :, self.keep :]))) if self.keep < n_xi else 0.0
        if tail != 0.0:
            raise ValueError(f"RHS has Legendre support at l >= {self.keep} (max {tail:.3e})")

        t_solve = 0.0
        f_low = np.zeros((n_s, n_x, self.keep, n_tz), dtype=np.float64)
        s_val = np.zeros((n_s, n_x), dtype=np.float64)
        for i_s in range(n_s):
            for i_x in range(n_x):
                rhs_low = jnp.stack(
                    [rhs_f[i_s, i_x, : self.keep], jnp.zeros((self.keep, n_tz)).at[0].set(self.b0)],
                    axis=2,
                )  # (keep, TZ, 2): the physical RHS and the border column z
                t0 = time.perf_counter()
                x_low = self._solve_one(
                    self.stream_s[i_s],
                    self.mirror_s[i_s],
                    self.pas_coef[i_s, i_x],
                    op.x[i_x],
                    self.gamma[i_s, i_x],
                    rhs_low,
                    self.n_blocks_per_x[i_x],
                )
                x_low.block_until_ready()
                t_solve += time.perf_counter() - t0
                y, z = np.asarray(x_low[:, :, 0]), np.asarray(x_low[:, :, 1])
                c0 = np.asarray(self.c0)
                shift = (c0 @ y[0] - r_c[i_s, i_x]) / (c0 @ z[0])
                s_val[i_s, i_x] = float(self.gamma[i_s, i_x]) * r_c[i_s, i_x] + shift
                f_low[i_s, i_x] = y - shift * z

        x_full = np.zeros((op.total_size,), dtype=np.float64)
        f_view = x_full[: op.f_size].reshape(n_s, n_x, n_xi, n_tz)
        f_view[:, :, : self.keep] = f_low
        x_full[op.f_size :] = s_val.reshape(-1)
        return x_full, {"solve_elim_s": t_solve}

    # -- partial residual check -------------------------------------------------
    def partial_residual(self, x_full: np.ndarray, rhs: np.ndarray) -> float:
        """Max relative residual over rows that involve only l < keep_lowest.

        Row l of the block-tridiagonal system touches columns l-1, l, l+1, so
        rows l = 0 .. keep-2 (plus the border/constraint rows) are fully
        determined by the computed solution blocks and can be verified exactly.
        """
        op = self.op
        n_s, n_x, n_xi, n_t, n_z = op.f_shape
        n_tz = n_t * n_z
        f = x_full[: op.f_size].reshape(n_s, n_x, n_xi, n_tz)
        s_val = x_full[op.f_size :].reshape(n_s, n_x)
        rhs_f = np.asarray(rhs[: op.f_size]).reshape(n_s, n_x, n_xi, n_tz)
        r_c = np.asarray(rhs[op.f_size :]).reshape(n_s, n_x)
        c0 = np.asarray(self.c0)
        b0 = np.asarray(self.b0)
        rhs_scale = max(float(np.linalg.norm(rhs)), 1e-300)  # global: l=1 blocks are zero
        worst = 0.0
        for i_s in range(n_s):
            for i_x in range(n_x):
                for ell in range(self.keep - 1):
                    lo, di, up = (
                        np.asarray(a)
                        for a in self._blocks_at(
                            jnp.asarray(ell, dtype=jnp.int32),
                            self.stream_s[i_s],
                            self.mirror_s[i_s],
                            self.pas_coef[i_s, i_x],
                            self.op.x[i_x],
                            jnp.asarray(0.0),  # raw block, no rank-one shift
                        )
                    )
                    r = di @ f[i_s, i_x, ell] - rhs_f[i_s, i_x, ell]
                    if ell > 0:
                        r += lo @ f[i_s, i_x, ell - 1]
                    if ell + 1 < self.n_blocks_per_x[i_x]:
                        r += up @ f[i_s, i_x, ell + 1]
                    if ell == 0:
                        r += b0 * s_val[i_s, i_x]
                    worst = max(worst, float(np.linalg.norm(r)) / rhs_scale)
                # constraint row: c0 . f_{l=0} = r_c
                worst = max(
                    worst, abs(float(c0 @ f[i_s, i_x, 0]) - r_c[i_s, i_x]) / rhs_scale
                )
        return worst


# =============================================================================
# Moments (the consolidated diagnostics formulas in dkx.moments)
# =============================================================================


def compute_moments(op: KineticOperator, x_full: np.ndarray) -> dict[str, np.ndarray]:
    layout = StateLayout(
        n_species=op.n_species, n_x=op.n_x, n_xi=op.n_xi, n_theta=op.n_theta,
        n_zeta=op.n_zeta, include_phi1=False, constraint_scheme=op.constraint_scheme,
    )  # fmt: skip
    vgrid = VelocityGrid(x=op.x, x_weights=op.x_weights, n_xi_for_x=op.n_xi_for_x)
    surface = FluxSurface.from_operator(op)
    species = SpeciesParams.from_operator(op)
    table = rhsmode1_moments(
        layout, vgrid, surface, species, jnp.asarray(x_full),
        delta=op.delta, alpha=op.alpha,
    )  # fmt: skip
    keys = ("FSABFlow", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat", "sources")
    return {k: np.asarray(table[k]) for k in keys}


# =============================================================================
# Small-grid cross-validation: truncated kernel vs full tier-1 factorization
# =============================================================================


def validate_small(base_text: str, n_xi: int = 30) -> None:
    from dkx.solve import solve as canonical_solve

    text = patch_namelist(
        base_text, "resolutionParameters", {"Ntheta": "13", "Nzeta": "25", "Nxi": str(n_xi)}
    )
    text = patch_namelist(text, "otherNumericalParameters", {"Nxi_for_x_option": "0"})
    nml = parse_sfincs_input_text(text, source_path="validate_small.namelist")
    op = KineticOperator.from_namelist(nml)
    print(f"[validate] small grid: shape={op.f_shape}, total_size={op.total_size}, "
          f"full bands = {full_band_gigabytes(op):.2f} GB")
    rhs = op.rhs()

    ref = canonical_solve(op, rhs, method="block_tridiagonal")
    x_ref = np.asarray(ref.x)
    print(f"[validate] full tier-1: residual={float(ref.residual_norms[0]):.3e}, "
          f"converged={ref.converged}")

    trunc = TruncatedTier1(op)
    x_tr, _ = trunc.solve(rhs)

    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    f_ref = x_ref[: op.f_size].reshape(n_s, n_x, n_xi, n_t * n_z)
    f_tr = x_tr[: op.f_size].reshape(n_s, n_x, n_xi, n_t * n_z)
    dl = np.linalg.norm(f_tr[:, :, :KEEP_LOWEST] - f_ref[:, :, :KEEP_LOWEST])
    nl = np.linalg.norm(f_ref[:, :, :KEEP_LOWEST])
    ds = np.linalg.norm(x_tr[op.f_size :] - x_ref[op.f_size :])
    ns = np.linalg.norm(x_ref[op.f_size :])
    print(f"[validate] truncated vs full: |df(l<3)|/|f(l<3)| = {dl / nl:.3e}, "
          f"|dsources|/|sources| = {ds / max(ns, 1e-300):.3e}")

    m_ref = compute_moments(op, x_ref)
    m_tr = compute_moments(op, x_tr)
    for k in ("FSABFlow", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat"):
        rel = np.abs(m_tr[k] - m_ref[k]) / np.maximum(np.abs(m_ref[k]), 1e-300)
        print(f"[validate] {k}: max rel diff = {rel.max():.3e}")


def ramp_effect_small(base_text: str, n_xi: int = 60) -> None:
    """Bound the Nxi_for_x ramp-vs-uniform physics effect on a reduced grid.

    Both discretizations are solved exactly with the truncated kernel (the
    ramp is handled with n_blocks = Nxi_for_x[ix] per closed subsystem) and
    the RHSMode=1 output moments are compared.
    """
    results = {}
    for label, option in (("uniform", "0"), ("ramp", "1")):
        text = patch_namelist(
            base_text, "resolutionParameters", {"Ntheta": "13", "Nzeta": "25", "Nxi": str(n_xi)}
        )
        text = patch_namelist(text, "otherNumericalParameters", {"Nxi_for_x_option": option})
        nml = parse_sfincs_input_text(text, source_path=f"ramp_{label}.namelist")
        op = KineticOperator.from_namelist(nml)
        print(f"[ramp:{label}] Nxi_for_x = {np.asarray(op.n_xi_for_x).tolist()}")
        rhs = op.rhs()
        trunc = TruncatedTier1(op)
        x_full, _ = trunc.solve(rhs)
        pres = trunc.partial_residual(x_full, np.asarray(rhs))
        print(f"[ramp:{label}] partial residual = {pres:.3e}")
        results[label] = compute_moments(op, x_full)
    for k in ("FSABFlow", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat"):
        a, b = results["uniform"][k], results["ramp"][k]
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
        print(f"[ramp] {k}: uniform={a.ravel()} ramp={b.ravel()} max rel diff={rel.max():.3e}")


# =============================================================================
# Main benchmark
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Fortran baseline input.namelist")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                    help="JAX backend (sets JAX_PLATFORMS before backend init)")
    ap.add_argument("--repeat", type=int, default=1, help="number of warm (post-compile) solves")
    ap.add_argument("--fortran-h5", type=Path, default=None, help="Fortran sfincsOutput.h5 to compare against")
    ap.add_argument("--fortran-solve-s", type=float, nargs=2, default=(463.6, 229.5),
                    metavar=("N1", "N2"), help="Fortran solve wall (1 rank, 2 ranks)")
    ap.add_argument("--fortran-rss-gb", type=float, nargs=2, default=(3.98, 2.86),
                    metavar=("N1", "N2"), help="Fortran peak RSS GB (1 rank, 2 ranks)")
    ap.add_argument("--memory-budget-gb", type=float, default=8.0,
                    help="route to the truncated kernel above this full tier-1 estimate")
    ap.add_argument("--validate-small", action="store_true",
                    help="only run the reduced-grid truncated-vs-full cross-check")
    ap.add_argument("--small-nxi", type=int, default=30,
                    help="Nxi for --validate-small (memory of the full reference "
                         "factorization scales linearly with it)")
    ap.add_argument("--ramp-effect", action="store_true",
                    help="only bound the Nxi_for_x ramp effect on a reduced grid")
    ap.add_argument("--ramp", action="store_true",
                    help="keep the namelist's Nxi_for_x ramp (option 1) instead of forcing "
                         "uniform Nxi_for_x — solves the exact Fortran discretization by "
                         "using n_blocks = Nxi_for_x[ix] per (species, x) subsystem")
    args = ap.parse_args()

    print(f"jax backend: {jax.default_backend()}, devices: {jax.devices()}")

    base_text = args.input.read_text()
    if args.validate_small:
        validate_small(base_text, n_xi=args.small_nxi)
        return
    if args.ramp_effect:
        ramp_effect_small(base_text)
        return

    # ---- 1. build the operator (default: uniform Nxi_for_x, strictly larger than
    # the Fortran ramped problem; --ramp keeps the namelist's ramp = Fortran's exact
    # discretization) ----
    if args.ramp:
        text = base_text
    else:
        text = patch_namelist(base_text, "otherNumericalParameters", {"Nxi_for_x_option": "0"})
    nml = parse_sfincs_input_text(text, source_path=str(args.input))
    t0 = time.perf_counter()
    op = KineticOperator.from_namelist(nml)
    jax.block_until_ready(op.b_hat)
    t_build = time.perf_counter() - t0
    n_tz = op.n_theta * op.n_zeta
    print(f"operator: shape (S,X,L,T,Z)={op.f_shape}, TZ={n_tz}, total_size={op.total_size}")
    print(f"Nxi_for_x = {np.asarray(op.n_xi_for_x).tolist()} "
          f"({'ramp (Fortran discretization)' if args.ramp else 'uniform (>= Fortran)'})")
    print(f"operator build: {t_build:.2f} s   [RSS {rss_gb():.2f} GB]")

    # ---- 2. route choice by memory estimate ----
    bands_gb = full_band_gigabytes(op)
    est_full_gb = 2.5 * bands_gb  # bands + factors + temporaries
    trunc_gb = (
        (KEEP_LOWEST * 4 + 8) * n_tz * n_tz * 8.0 / 2**30  # head stacks + scan temporaries
        + 3.0 * n_tz * n_tz * 8.0 / 2**30  # coefficient matrices (stream x2, exb)
    )
    uniform = int(np.min(np.asarray(op.n_xi_for_x))) == op.n_xi
    use_full = est_full_gb <= args.memory_budget_gb and uniform and not args.ramp
    print(f"memory estimate: full tier-1 ~{est_full_gb:.1f} GB, truncated ~{trunc_gb:.2f} GB "
          f"(budget {args.memory_budget_gb} GB)")
    print("route: " + ("canonical solve.py tier-1 (fits budget)" if use_full else
                       "truncated block-Thomas (block_thomas_truncated_fn), keep_lowest=3"))

    # ---- 3. RHS + solver setup ----
    t0 = time.perf_counter()
    rhs = op.rhs()
    if use_full:
        from dkx.solve import solve as canonical_solve

        def do_solve():
            res = canonical_solve(op, rhs, method="block_tridiagonal")
            jax.block_until_ready(res.x)
            return np.asarray(res.x), {"solve_elim_s": float("nan")}

        jax.block_until_ready(rhs)
        solver = None
    else:
        solver = TruncatedTier1(op)
        jax.block_until_ready(solver.stream_s)

        def do_solve():
            return solver.solve(rhs)

    t_assemble = time.perf_counter() - t0
    print(f"RHS + coefficient assembly: {t_assemble:.2f} s   [RSS {rss_gb():.2f} GB]")
    g = gpu_mem_gb()
    if g is not None:
        print(f"GPU memory after assembly: {g:.2f} GB")

    # ---- 4. cold and warm solves ----
    t0 = time.perf_counter()
    x_full, info = do_solve()
    t_cold = time.perf_counter() - t0
    print(f"cold solve (with jit compile): {t_cold:.2f} s "
          f"(elimination {info['solve_elim_s']:.2f} s)   [RSS {rss_gb():.2f} GB]")
    g = gpu_mem_gb()
    if g is not None:
        print(f"GPU memory after cold solve: {g:.2f} GB")

    warm_times: list[float] = []
    for _ in range(max(args.repeat, 0)):
        t0 = time.perf_counter()
        x_full, info = do_solve()
        warm_times.append(time.perf_counter() - t0)
        print(f"warm solve: {warm_times[-1]:.2f} s (elimination {info['solve_elim_s']:.2f} s)   "
              f"[RSS {rss_gb():.2f} GB]")
    t_warm = min(warm_times) if warm_times else float("nan")

    # ---- 5. exactness check ----
    t0 = time.perf_counter()
    if solver is not None:
        pres = solver.partial_residual(x_full, np.asarray(rhs))
        print(f"partial residual (rows l<{KEEP_LOWEST - 1} + constraints): {pres:.3e} "
              f"({time.perf_counter() - t0:.2f} s)")
    else:
        r = np.asarray(op.apply(jnp.asarray(x_full))) - np.asarray(rhs)
        print(f"full residual: {np.linalg.norm(r) / np.linalg.norm(np.asarray(rhs)):.3e} "
              f"({time.perf_counter() - t0:.2f} s)")

    # ---- 6. output moments ----
    t0 = time.perf_counter()
    moments = compute_moments(op, x_full)
    t_mom = time.perf_counter() - t0
    print(f"moments: {t_mom:.2f} s   [RSS {rss_gb():.2f} GB]")

    # ---- 7. compare against Fortran ----
    print()
    print("=" * 88)
    print("RESULTS — HSX PAS DKES RHSMode=1  (Ntheta=25 Nzeta=51 Nxi=100 Nx=5; 744,610 unknowns)")
    print("=" * 88)
    f1, f2 = args.fortran_solve_s
    r1, r2 = args.fortran_rss_gb
    print(f"{'':28s}{'JAX (this run)':>18s}{'Fortran 1 rank':>18s}{'Fortran 2 ranks':>18s}")
    print(f"{'solve wall, cold [s]':28s}{t_cold:>18.1f}{f1:>18.1f}{f2:>18.1f}")
    print(f"{'solve wall, warm [s]':28s}{t_warm:>18.1f}{'—':>18s}{'—':>18s}")
    print(f"{'peak RSS so far [GB]':28s}{rss_gb():>18.2f}{r1:>18.2f}{r2:>18.2f}")
    print(f"{'operator build [s]':28s}{t_build:>18.2f}")
    print(f"{'coefficient assembly [s]':28s}{t_assemble:>18.2f}")
    print(f"{'moments [s]':28s}{t_mom:>18.2f}")
    print("(definitive peak RSS: run this script under /usr/bin/time -l (macOS) or -v (Linux))")

    if args.fortran_h5 is not None:
        import h5py

        with h5py.File(args.fortran_h5, "r") as h5:
            print()
            print(f"{'quantity':34s}{'species':>8s}{'JAX':>16s}{'Fortran':>16s}{'rel diff':>12s}")
            for key in ("FSABFlow", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat"):
                ref = np.asarray(h5[key][...]).reshape(op.n_species, -1)[:, -1]  # last iteration
                val = moments[key].reshape(-1)
                for i_s in range(op.n_species):
                    rel = abs(val[i_s] - ref[i_s]) / max(abs(ref[i_s]), 1e-300)
                    print(f"{key:34s}{i_s:>8d}{val[i_s]:>16.8e}{ref[i_s]:>16.8e}{rel:>12.2e}")
            print()
            if not args.ramp:
                print("note: Fortran uses the Nxi_for_x ramp (option 1); JAX solved the strictly")
                print("larger uniform-Nxi_for_x problem, so sub-1e-3 agreement is discretization-level.")


if __name__ == "__main__":
    main()
