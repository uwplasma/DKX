"""AD-vs-FD gradient verification table for the methods paper.

Three physics-relevant derivatives, each computed twice on the same tiny
deck: once with JAX automatic differentiation through the full solve
(implicit differentiation of the linear or root-finding problem, via
``jax.grad``), and once with central finite differences of the same scalar
objective.  The rows exercise the three differentiable tiers of the code:

  (a) ``d(D11*)/d(B_10)``: derivative of the normalized monoenergetic
      radial-diffusion coefficient (Beidler et al. normalization) with
      respect to a Boozer ``|B|`` cosine amplitude, through the
      RHSMode=3 monoenergetic-database path
      (:func:`sfincs_jax.monoenergetic.monoenergetic_database_from_operator`
      with ``differentiable=True`` solves on a
      :meth:`~sfincs_jax.magnetic_geometry.FluxSurfaceGeometry.from_fourier`
      geometry) -- the geometry-optimization entry point.

  (b) ``d(FSABjHat)/d(nu_n)``: derivative of the flux-surface-averaged
      bootstrap current of a two-species RHSMode=1 solve with respect to
      the normalized collision frequency, through the implicit-diff linear
      solve (:func:`sfincs_jax.solve.solve` with ``differentiable=True``)
      and the moment assembly (:func:`sfincs_jax.moments.rhsmode1_moments`).

  (c) ``d(Er_root)/d(theta_n)``: derivative of the ambipolar radial
      electric field with respect to a scale factor ``theta_n`` on both
      species' density gradients, through the implicit-function-theorem
      root solve (:func:`sfincs_jax.er.ambipolar_er`, ``jax.lax.custom_root``
      via solvax) -- no finite differences anywhere in the AD path.

Outputs: ``docs/_static/figures/paper_benchmarks/gradient_verification.json``
plus an rst ``list-table`` snippet next to it (not yet wired into any docs
page), and the table printed to stdout.

Expected runtime: ~2 min on a laptop CPU (all rows use tiny analytic-
geometry decks; each row also runs two finite-difference re-solves).

Run (from the repo root):
  python examples/paper_benchmarks/gradient_verification.py
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sfincs_jax import er as er_mod  # noqa: E402
from sfincs_jax.drift_kinetic import (  # noqa: E402
    _geometry_and_radial,
    kinetic_operator_from_namelist,
)
from sfincs_jax.inputs import load_sfincs_input  # noqa: E402
from sfincs_jax.magnetic_geometry import FluxSurfaceGeometry  # noqa: E402
from sfincs_jax.moments import rhsmode1_moments  # noqa: E402
from sfincs_jax.monoenergetic import monoenergetic_database_from_operator  # noqa: E402
from sfincs_jax.run import _grids_from_input, _raw_with_validated_overrides  # noqa: E402
from sfincs_jax.solve import solve  # noqa: E402
from sfincs_jax.writer import operator_containers  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "output"
FIG_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks"
JSON_PATH = FIG_DIR / "gradient_verification.json"
RST_PATH = FIG_DIR / "gradient_verification.rst"

# ----------------------------------------------------------------------------
# Tiny decks (analytic geometry; no external files)
# ----------------------------------------------------------------------------

# Row (a): single-species monoenergetic deck on a three-helicity analytic
# Boozer field (scheme-1-like spectrum rebuilt through from_fourier so the
# amplitudes are traced).
MONO_DECK = """&general
  RHSMode = 3
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.5
  epsilon_t = -0.07053
  epsilon_h = 0.05067
  epsilon_antisymm = 0.0
  iota = 0.4542
  GHat = 3.7481
  IHat = 0.0
  B0OverBBar = 1.0
  helicity_l = 2
  helicity_n = 10
  psiAHat = 0.15596
  aHat = 0.5585
/
&speciesParameters
/
&physicsParameters
  nuPrime = 0.3
  EStar = 0.0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 13
  Nzeta = 27
  Nxi = 24
  Nx = 1
  solverTolerance = 1d-10
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
"""

# Rows (b) and (c): two-species PAS deck on a helically rippled analytic
# field with an ambipolar ion root (the er-module regression geometry).
PAS_DECK = """&general
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0
  GHat = 1.0
  IHat = 0.0
  iota = 1.31
  epsilon_t = 0.1
  epsilon_h = 0.05
  helicity_l = 2
  helicity_n = 5
  psiAHat = 0.045
  aHat = 0.1
/
&speciesParameters
  Zs = 1 -1
  mHats = 1.0 0.000545509
  nHats = 1.0 1.0
  THats = 1.0 1.0
  dNHatdrHats = -0.5 -0.5
  dTHatdrHats = -1.0 -1.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = 8.4774d-3
  Er = 0.0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 7
  Nzeta = 7
  Nxi = 8
  NL = 4
  Nx = 3
  solverTolerance = 1d-10
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# ----------------------------------------------------------------------------
# Row (a): d(D11*)/d(B_10) through the monoenergetic-database path
# ----------------------------------------------------------------------------
def row_d11_geometry(tmp: Path) -> dict:
    inp = load_sfincs_input(_write(tmp / "mono.input.namelist", MONO_DECK))
    raw = _raw_with_validated_overrides(inp)
    grids = _grids_from_input(inp, raw)
    geom, radial = _geometry_and_radial(nml=raw, grids=grids)
    op0 = kinetic_operator_from_namelist(raw)

    g_hat, i_hat = float(geom.g_hat), float(geom.i_hat)
    iota, b0 = float(geom.iota), float(geom.b0_over_bbar)
    r_hat = float(np.sqrt(2.0 * abs(float(radial.psi_hat)) / abs(b0)))
    theta = jnp.asarray(grids.theta)
    zeta = jnp.asarray(grids.zeta)
    # The scheme-1 spectrum: (0,0), the helical (2,1) term, and the traced
    # "mirror-like" (1,0) amplitude (epsilon_t at rN = 0.5 is -0.03527).
    m_modes = jnp.asarray([0, 1, 2])
    n_modes = jnp.asarray([0, 0, 1])
    amp_helical = 0.05067 * 0.25  # epsilon_h * rN^2

    def d11_of(amp: jnp.ndarray) -> jnp.ndarray:
        bmnc = jnp.stack([jnp.asarray(1.0), amp, jnp.asarray(amp_helical)])
        fourier = FluxSurfaceGeometry.from_fourier(
            theta=theta, zeta=zeta, bmnc=bmnc, m=m_modes, n=n_modes,
            n_periods=10, iota=iota, g_hat=g_hat, i_hat=i_hat,
        )  # fmt: skip
        op_traced = replace(
            op0,
            b_hat=fourier.b_hat,
            db_hat_dtheta=fourier.db_hat_dtheta,
            db_hat_dzeta=fourier.db_hat_dzeta,
            d_hat=fourier.d_hat,
            b_hat_sup_theta=fourier.b_hat_sup_theta,
            b_hat_sup_zeta=fourier.b_hat_sup_zeta,
            b_hat_sub_theta=fourier.b_hat_sub_theta,
            b_hat_sub_zeta=fourier.b_hat_sub_zeta,
            fsab_hat2=fourier.fsab_hat2(
                theta_weights=op0.theta_weights, zeta_weights=op0.zeta_weights
            ),
        )
        db = monoenergetic_database_from_operator(
            op_traced, [0.3], (0.0,),
            g_hat=g_hat, i_hat=i_hat, iota=iota, b0_over_bbar=b0, r_hat=r_hat,
            differentiable=True,
        )  # fmt: skip
        return db.d11_star[0, 0]

    amp0 = jnp.asarray(-0.07053 * 0.5)  # epsilon_t * rN
    t0 = time.time()
    grad_ad = float(jax.grad(d11_of)(amp0))
    ad_seconds = time.time() - t0
    eps = 1e-5
    t0 = time.time()
    grad_fd = float((d11_of(amp0 + eps) - d11_of(amp0 - eps)) / (2.0 * eps))
    fd_seconds = time.time() - t0
    return {
        "objective": "D11* (nuPrime=0.3, EStar=0)",
        "parameter": "Boozer amplitude B_10 (mirror-like term)",
        "path": "RHSMode=3 monoenergetic database, from_fourier geometry, implicit-diff solve",
        "value": float(d11_of(amp0)),
        "ad": grad_ad,
        "fd": grad_fd,
        "fd_step": eps,
        "rel_dev": abs(grad_ad - grad_fd) / abs(grad_fd),
        "ad_seconds": ad_seconds,
        "fd_seconds": fd_seconds,
    }


# ----------------------------------------------------------------------------
# Row (b): d(FSABjHat)/d(nu_n) through a RHSMode=1 solve
# ----------------------------------------------------------------------------
def row_fsabjhat_nu_n(tmp: Path) -> dict:
    inp = load_sfincs_input(_write(tmp / "pas.input.namelist", PAS_DECK))
    op0 = kinetic_operator_from_namelist(inp.raw)
    nu0 = jnp.asarray(op0.pas.nu_n, dtype=jnp.float64)

    def fsabjhat(nu_n: jnp.ndarray) -> jnp.ndarray:
        # The PAS coefficient table is linear in nu_n: rescale in place.
        pas = replace(op0.pas, nu_n=nu_n, coef=op0.pas.coef * (nu_n / op0.pas.nu_n))
        op = replace(op0, pas=pas)
        x = solve(op, op.rhs(), tol=1e-12, differentiable=True).x
        layout, vgrid, surface, species = operator_containers(op)
        table = rhsmode1_moments(
            layout, vgrid, surface, species, jnp.reshape(x, (-1,)),
            delta=op.delta, alpha=op.alpha,
        )  # fmt: skip
        # FSABjHat = sum_s Z_s <B Flow_s> (species-summed bootstrap current).
        return jnp.reshape(table["FSABjHat"], ())

    t0 = time.time()
    grad_ad = float(jax.grad(fsabjhat)(nu0))
    ad_seconds = time.time() - t0
    h = 1e-5 * float(nu0)
    t0 = time.time()
    grad_fd = float((fsabjhat(nu0 + h) - fsabjhat(nu0 - h)) / (2.0 * h))
    fd_seconds = time.time() - t0
    return {
        "objective": "FSABjHat (two-species RHSMode=1 solve)",
        "parameter": "nu_n (normalized collision frequency)",
        "path": "RHSMode=1 linear solve (implicit diff) + moment assembly",
        "value": float(fsabjhat(nu0)),
        "ad": grad_ad,
        "fd": grad_fd,
        "fd_step": h,
        "rel_dev": abs(grad_ad - grad_fd) / abs(grad_fd),
        "ad_seconds": ad_seconds,
        "fd_seconds": fd_seconds,
    }


# ----------------------------------------------------------------------------
# Row (c): d(ambipolar Er)/d(density-gradient scale) through the root solve
# ----------------------------------------------------------------------------
def row_ambipolar_er(tmp: Path) -> dict:
    deck = _write(tmp / "er.input.namelist", PAS_DECK)
    prob = er_mod.prepare(deck, er_bracket=(-3.0, 1.0))
    found = er_mod.find_ambipolar_er(deck, er_bracket=(-3.0, 1.0), all_roots=False, emit=None)
    root = float(found.er)
    base_op = prob.operator
    base_dn = base_op.dn_hat_dpsi_hat

    def er_of(theta_n):
        op_theta = replace(base_op, dn_hat_dpsi_hat=base_dn * theta_n)
        return er_mod.ambipolar_er(op_theta, er0=root, dphi_per_er=prob.dphi_per_er, z_s=prob.z_s)

    t0 = time.time()
    grad_ad = float(jax.grad(er_of)(1.0))
    ad_seconds = time.time() - t0
    h = 1e-3
    t0 = time.time()
    grad_fd = float((er_of(1.0 + h) - er_of(1.0 - h)) / (2.0 * h))
    fd_seconds = time.time() - t0
    return {
        "objective": "ambipolar Er root (ion root, two-species PAS deck)",
        "parameter": "theta_n (scale on both species' dnHat/dpsiHat)",
        "path": "implicit-function-theorem root solve (jax.lax.custom_root)",
        "value": float(er_of(1.0)),
        "ad": grad_ad,
        "fd": grad_fd,
        "fd_step": h,
        "rel_dev": abs(grad_ad - grad_fd) / abs(grad_fd),
        "ad_seconds": ad_seconds,
        "fd_seconds": fd_seconds,
    }


ROWS = [
    ("a", row_d11_geometry),
    ("b", row_fsabjhat_nu_n),
    ("c", row_ambipolar_er),
]


def main() -> None:
    print("=== examples/paper_benchmarks/gradient_verification.py ===", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "gradient_verification_decks"
    tmp.mkdir(parents=True, exist_ok=True)

    records = []
    for tag, fn in ROWS:
        t0 = time.time()
        rec = {"row": tag, **fn(tmp)}
        rec["row_seconds"] = time.time() - t0
        records.append(rec)
        print(
            f"  ({tag}) {rec['objective']}\n"
            f"      d/d[{rec['parameter']}]\n"
            f"      AD = {rec['ad']:+.10e}   FD = {rec['fd']:+.10e}   "
            f"rel dev = {rec['rel_dev']:.2e}   [{rec['row_seconds']:.1f} s]",
            flush=True,
        )

    record = {
        "table": "AD-vs-FD gradient verification (central finite differences)",
        "note": (
            "Each row evaluates the same scalar objective twice: jax.grad "
            "through the full solve (implicit differentiation), and a "
            "2-point central finite difference with the step listed.  "
            "Tiny analytic-geometry decks; float64."
        ),
        "rows": records,
    }
    JSON_PATH.write_text(json.dumps(record, indent=1))
    print(f"  wrote {JSON_PATH}")

    lines = [
        ".. list-table:: AD-vs-FD gradient verification",
        "   :header-rows: 1",
        "",
        "   * - Objective",
        "     - Parameter",
        "     - AD",
        "     - Central FD",
        "     - Rel. deviation",
    ]
    for rec in records:
        lines += [
            f"   * - {rec['objective']}",
            f"     - {rec['parameter']}",
            f"     - {rec['ad']:+.8e}",
            f"     - {rec['fd']:+.8e}",
            f"     - {rec['rel_dev']:.1e}",
        ]
    RST_PATH.write_text("\n".join(lines) + "\n")
    print(f"  wrote {RST_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
