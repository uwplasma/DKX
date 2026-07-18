"""Flagship optimization: a quasi-axisymmetric stellarator with low bootstrap current.

What this example teaches:
  - the modern two-stage stellarator design loop that reaches a *genuine*
    quasi-axisymmetric (QA) equilibrium and then lowers its bootstrap current:

      Stage A -- QA shaping (vmex.optimize.least_squares).  Starting from a
        circular torus (input.minimal_seed_nfp2, R0 = 1 m, a = 0.2 m, exactly
        axisymmetric so its rotational transform vanishes at first order), a
        staged max_mode continuation drives the two-term quasisymmetry ratio
        residual to zero while holding aspect ratio at 6 and mean iota at 0.42.
        The decision variables are the boundary Fourier coefficients RBC/ZBS;
        the gradients are the exact implicit (adjoint) Jacobian of the
        fixed-boundary equilibrium (jac="implicit"), no finite differences.
        This is the reference vmex QA recipe.

      Stage B -- bootstrap reduction at HELD precise QA (this is where
        dkx enters).  The QA equilibrium from stage A is quasisymmetric
        but was never optimized for its neoclassical bootstrap current, so
        there is real headroom.  A gradient-based loop minimizes
        <j.B>/sqrt(<B^2>) -- computed by the canonical dkx kinetic
        solve -- while a hard one-sided cap holds the two-term quasisymmetry
        ratio residual (the very metric stage A minimized) at the Stage-A
        precise-QA level, at aspect ratio 6 and mean iota above 0.41.  Both
        compared configurations are therefore precise QA: the showcase is
        precise QA (no bootstrap optimization) vs precise QA held + bootstrap
        optimization, so the reported bootstrap decrease is the reduction
        achievable at fixed quasisymmetry -- a smaller, honest factor than a
        QA-degrading search would advertise.

  - the differentiable route between the codes used by stage B, so that one
    jax.value_and_grad call returns the gradient of the *whole* physics chain:
        boundary dofs -> vmex.core.implicit.solve_implicit (fixed-boundary
        MHD equilibrium with an implicit-adjoint custom VJP)
        -> traceable single-surface VMEC spectral tables
           (vmex.core.boozer_tables.boozer_input_tables; validated
           against the host wout tables in tests/test_example_qa_bootstrap.py)
        -> booz_xform_jax (differentiable Boozer transform, |B| spectrum)
        -> FluxSurfaceGeometry.from_fourier (geometryScheme-13 pure-JAX path)
        -> KineticOperator -> tier-2 GCROT solve with implicit differentiation
        -> FSABjHat;

  - a constrained physics target expressed as penalty terms: hold the field
    quasisymmetric at the Stage-A level (a hard one-sided cap on the two-term
    quasisymmetry ratio residual across the volume, plus the Boozer-spectrum QA
    metric -- the energy fraction of the symmetry-breaking n != 0 modes of |B|
    -- on the kinetic surface), hold aspect ratio at 6, keep mean iota above
    0.41 (one-sided hinge), and drive <j.B> toward zero;

  - warm-starting the kinetic Krylov solve across optimizer evaluations with
    the previous solution (x0) and the GCROT recycle pair, hot-restarting the
    host VMEC solve from the previous boundary (make_config(hot_restart=True))
    and using a loose equilibrium adjoint tolerance (the trust region only
    needs ~1e-3 gradients), so every warm evaluation is a few seconds; and

  - verifying the end-to-end gradient against central finite differences at
    the starting point AND at the optimized end point.

Physics: quasi-axisymmetry makes the guiding-centre drifts tokamak-like, so a
QA field carries a substantial bootstrap current -- lowering it at fixed
quasisymmetry, aspect ratio and iota is a genuine Pareto trade, not a free
lunch (holding QA fixed makes the achievable bootstrap decrease honestly
smaller than a search that is free to spend quasisymmetry).  The kinetic
configuration is the classic SFINCS "full trajectories" setup: two-species
pitch-angle-scattering collisions with a finite radial electric field
(includeXDotTerm and includeElectricFieldTermInXiDot on), which routes to the
tier-2 GCROT solver where warm starts and recycling matter [Landreman, Smith,
Mollen & Helander, Phys. Plasmas 21, 042503 (2014); Landreman & Paul, PRL 128,
035001 (2022)].

Gradient accuracy (measured, documented honestly):
  - Boozer-spectrum -> kinetic <j.B> segment: autodiff vs central FD agree to
    ~3e-6 relative (pure JAX + implicit linear solve).
  - full chain d(objective)/d(boundary dof): the dominant dof agrees with
    central FD to ~7e-3 at the default resolution (looser at CI resolution).
    An FD-step sweep shows the FD value converging monotonically TOWARD the
    stable autodiff value and plateauing at the host equilibrium solver's
    ftol termination-noise floor -- the comparison is limited by
    finite differences, not by the autodiff chain.
  - CPU vs GPU: the objective agrees to ~8e-11 relative at identical inputs.

Expected runtime: stage A is dominated by the one-time implicit-Jacobian XLA
compile per continuation stage (warm forward solves ~1 s); stage B by the host
VMEC solve plus the kinetic value_and_grad (a few seconds per warm
evaluation).  The whole example runs in well under an hour on a laptop CPU.
Progress is appended to a per-evaluation log file and the best point is
checkpointed after every evaluation, so a long run is inspectable while it goes
and resumable (DKX_QA_DOFS_INIT).  With DKX_CI=1 everything
shrinks to a couple of minutes.

Requires the optional companions of this example (not needed by dkx
itself):  pip install -e /path/to/vmex /path/to/booz_xform_jax

Run:
  python examples/optimize_QA_bootstrap.py
"""

import dataclasses
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import scipy.optimize

from dkx.drift_kinetic import kinetic_operator_from_namelist
from dkx.inputs import parse_sfincs_input_text
from dkx.magnetic_geometry import FluxSurfaceGeometry
from dkx.phase_space import make_grids
from dkx.run import profile_moments_from_operator
from dkx.solve import solve as kinetic_solve

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

try:  # optional companion packages (not needed by dkx itself)
    import vmex as _vmex_pkg
    from vmex import optimize as vmec_optimize
    from vmex.core import implicit as vmec_implicit
    from vmex.core import solver as vmec_solver
    from vmex.core.boozer_tables import boozer_input_tables
    from vmex.core.input import VmecInput
    from vmex.core.wout import wout_from_state, write_wout
    from booz_xform_jax.jax_api import booz_xform_jax as booz_transform
except ImportError as exc:
    raise SystemExit(
        "This example needs vmex (new core API, with core.boozer_tables and "
        "optimize.least_squares) and booz_xform_jax. Install with "
        "`pip install -e /path/to/vmex /path/to/booz_xform_jax`."
    ) from exc

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
CI = os.environ.get("DKX_CI") == "1"  # shrink resolution for CI

# Stage-A seed: a circular torus shipped with vmex (examples/data of an
# editable checkout); resolved from the installed package so no sibling-
# directory layout is assumed.
SEED_INPUT = (
    Path(_vmex_pkg.__file__).resolve().parents[1]
    / "examples" / "data" / "input.minimal_seed_nfp2"
)
SEED_INPUT = Path(os.environ.get("DKX_QA_SEED_INPUT", SEED_INPUT))
# A small helical RBC/ZBS(n=1, m=1) kick breaks the circular torus' iota
# saddle (its transform is second-order in the 3D shaping, so its gradient
# vanishes there) -- the same tie-break the vmex QA example uses.
SEED_PERTURBATION = 0.03

# Equilibrium resolution and convergence (the host VMEC solves dominate cost).
NS_SHAPE = 7 if CI else 25  # stage-A shaping radial surfaces
NS = 7 if CI else 13        # stage-B kinetic radial surfaces
VMEC_FTOL = 1e-11 if CI else 1e-13
VMEC_MAX_ITER = 5000

# Boundary degrees of freedom: RBC/ZBS modes with m,|n| <= MAX_MODE
# (RBC(0,0), the major radius, stays fixed - same convention as simsopt).
MAX_MODE = 1 if CI else 2
SHAPE_SCHEDULE = (1,) if CI else (1, 2)  # stage-A max_mode continuation
SHAPE_MAX_NFEV = 3 if CI else 60         # trial budget per stage-A stage
SHAPE_FTOL = 1e-3 if CI else 1e-5

# Boozer transform resolution for the kinetic flux surface.
MBOZ, NBOZ = (2, 2) if CI else (4, 4)

# Kinetic solve: two species (ions + electrons), pitch-angle-scattering
# collisions, finite Er with the full-trajectory Er terms (this is what makes
# the system route to the tier-2 GCROT solver, where warm starts/recycling
# apply; it is also a nonsingular, exactly implicit-differentiable system).
KIN_NTHETA, KIN_NZETA, KIN_NXI, KIN_NL, KIN_NX = (9, 7, 8, 4, 4) if CI else (13, 11, 16, 4, 6)
ER = 1.0  # normalized radial electric field (template rHat convention)
NU_N = 0.1  # normalized collisionality
KINETIC_SOLVER_TOL = 1e-9

# Physics targets and penalty weights for the stage-B scalar objective.  The
# aspect target is two-sided; the iota constraint is the one-sided hinge
# max(0, IOTA_PEN_FLOOR - iota)^2, which is exactly zero while iota stays above
# IOTA_PEN_FLOOR and pushes back once it dips -- the margin above the hard
# requirement IOTA_MIN keeps the accepted iota safely > 0.41.
TARGET_ASPECT = 6.0
TARGET_IOTA = 0.42  # stage-A mean-iota target
IOTA_MIN = 0.41  # hard requirement: mean iota must end above this
IOTA_PEN_FLOOR = 0.415  # stage-B hinge activates below this (margin above IOTA_MIN)
QS_SURFACES = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])  # two-term QS guard
# Weights balance the Pareto trade between lowering <j.B> and *holding* QA at
# the Stage-A level: the kinetic weight lets the bootstrap term drive the
# search, the aspect weight pins aspect near 6, W_QS_BOOZ keeps the kinetic
# surface quasi-axisymmetric, and the hard one-sided cap W_QS_HOLD forbids the
# two-term ratio residual from rising past QS_HELD_TARGET -- so both compared
# configurations stay precise QA and the bootstrap decrease is measured at
# (essentially) fixed quasisymmetry.
W_ASPECT = 1.0e3
W_IOTA = 2.0e4
W_QS_BOOZ = 5.0e4  # Boozer-spectrum QA metric on the kinetic surface
W_QS_PROFILE = 5.0e3  # gentle pull of the two-term ratio residual toward zero
W_KINETIC = 5.0e6
# QA hold: a hard one-sided quadratic cap max(0, qs_profile - QS_HELD_TARGET)^2.
# QS_HELD_TARGET is set from the Stage-A residual after the first evaluation to
# QS_HELD_SLACK x that level; the slack is the small, documented QA budget the
# bootstrap search may spend (both configs stay precise QA, sub-1e-3 residual).
W_QS_HOLD = 1.0e10
QS_HELD_SLACK = 1.8
QS_HELD_TARGET = float("inf")  # set after the first (Stage-A) evaluation below

# Kinetic figure of merit entering the objective.  Every entry of the dict is
# CI-tested, so switching the commented line in just works.
KINETIC_OBJECTIVE = "bootstrap_jbs2"  # (<j.B>/sqrt(<B^2>))^2 -> drive to zero
# KINETIC_OBJECTIVE = "particle_flux_l1"  # tested: uncomment to use (L1-style smooth |Gamma_s| sum)
# KINETIC_OBJECTIVE = "heat_flux_l2"      # tested: uncomment to use (sum_s Q_s^2)

# Stage-B optimizer (scipy L-BFGS-B on jax.value_and_grad of the objective).
MAXITER = int(os.environ.get("DKX_QA_MAXITER", "2" if CI else "25"))
BOUND_RADIUS = 0.05  # box bounds |dof - dof0| <= radius keep trial boundaries physical
PENALTY_VALUE = 1.0e6  # returned for trial boundaries where VMEC fails (zero-crash)
RUN_FD_CHECK = os.environ.get("DKX_QA_FD_CHECK", "1") == "1"
# Central-FD step and acceptance gate for the end-to-end gradient check.  The
# autodiff gradient is exact (the Boozer->kinetic segment agrees with FD to
# ~3e-6, gated in the CI test); the *full-chain* FD comparison is limited by
# the host equilibrium solver's ftol termination noise.  An eps sweep
# (1e-5 .. 1e-3) shows the FD value converging monotonically toward the stable
# autodiff value and plateauing near ~7e-3 (the noise floor), so the gate below
# accommodates that floor and FD_EPS sits at the sweep optimum -- the comparison
# is limited by finite differences, not by the autodiff chain.
FD_EPS = 1e-5 if CI else 3e-4
FD_GATE = 5e-2 if CI else 1.5e-2

# Optional resume: point DKX_QA_DOFS_INIT at a checkpoint .npz written
# by a previous (interrupted) run to start stage B from its best point.
DOFS_INIT = os.environ.get("DKX_QA_DOFS_INIT", "")

OUT_DIR = Path(os.environ.get("DKX_QA_OUT_DIR", str(Path(__file__).parent / "output")))
STEM = "optimize_QA_bootstrap"

# ----------------------------------------------------------------------------
# 1) Stage A: circular torus -> precise QA via vmex.optimize.least_squares
# ----------------------------------------------------------------------------
print("=== examples/optimize_QA_bootstrap.py ===")
print("Stage A: QA shaping with vmex.optimize.least_squares (implicit Jacobian)")
if not SEED_INPUT.exists():
    raise SystemExit(
        f"seed input not found: {SEED_INPUT}\n"
        "Point DKX_QA_SEED_INPUT at input.minimal_seed_nfp2 from the "
        "vmex examples/data directory."
    )
seed = VmecInput.from_file(str(SEED_INPUT))
_rbc, _zbs = seed.rbc.copy(), seed.zbs.copy()
_rbc[seed.ntor + 1, 1] += SEED_PERTURBATION
_zbs[seed.ntor + 1, 1] += SEED_PERTURBATION
seed = dataclasses.replace(
    seed, rbc=_rbc, zbs=_zbs,
    ns_array=np.asarray([NS_SHAPE]),
    ftol_array=np.asarray([1e-12 if not CI else 1e-11]),
    niter_array=np.asarray([VMEC_MAX_ITER]),
)
NFP = int(seed.nfp)
qs_ratio = vmec_optimize.QuasisymmetryRatioResidual(
    np.linspace(0.1, 1.0, 10), helicity_m=1, helicity_n=0)
shaping_terms = [
    (qs_ratio, 0.0, 1.0),
    (vmec_optimize.aspect_ratio, TARGET_ASPECT, 1.0),
    (vmec_optimize.mean_iota, TARGET_IOTA, 10.0),
]


def _shape_report(tag, eq):
    total = float(qs_ratio.total(eq))
    aspect = float(vmec_optimize.aspect_ratio(eq.state, eq.runtime))
    iota = float(vmec_optimize.mean_iota(eq.state, eq.runtime))
    print(f"  [{tag}] QS ratio residual = {total:.4e}, aspect = {aspect:.4f}, "
          f"mean iota = {iota:.4f}")
    return total, aspect, iota


t_shape0 = time.perf_counter()
eq_seed = vmec_optimize.solve_equilibrium(seed)
qs_seed, aspect_seed, iota_seed = _shape_report("circular seed", eq_seed)
inp_shaped = seed
shape_result = None
for _mm in SHAPE_SCHEDULE:
    _ndof = len(vmec_optimize.boundary_dof_names(inp_shaped, _mm))
    print(f"  stage max_mode = {_mm} ({_ndof} boundary dofs)")
    shape_result = vmec_optimize.least_squares(
        shaping_terms, inp_shaped, max_mode=_mm, jac="implicit", use_ess=True,
        verbose=1, max_nfev=SHAPE_MAX_NFEV, ftol=SHAPE_FTOL, xtol=1e-9,
    )
    inp_shaped = shape_result.input
    if shape_result.equilibrium is not None:
        _shape_report(f"stage {_mm}", shape_result.equilibrium)
eq_shaped = (shape_result.equilibrium if shape_result is not None
             and shape_result.equilibrium is not None
             else vmec_optimize.solve_equilibrium(inp_shaped))
qs_shaped, aspect_shaped, iota_shaped = _shape_report("QA shaped", eq_shaped)
t_shape = time.perf_counter() - t_shape0
print(f"  stage A wall time: {t_shape:.1f} s   "
      f"(QS {qs_seed:.2e} -> {qs_shaped:.2e}, iota {iota_seed:.3f} -> {iota_shaped:.3f})")

# The shaped QA boundary is the starting point of the kinetic bootstrap
# reduction below; re-solve it at the (finer/coarser) kinetic radial mesh.
inp0 = dataclasses.replace(
    inp_shaped,
    ns_array=np.asarray([NS]),
    ftol_array=np.asarray([VMEC_FTOL]),
    niter_array=np.asarray([VMEC_MAX_ITER]),
)

# ----------------------------------------------------------------------------
# 2) Stage B setup: differentiable equilibrium + kinetic operator template
# ----------------------------------------------------------------------------
print("Stage B: reduce the kinetic bootstrap current <j.B> at HELD precise QA")
print("Step 1: differentiable fixed-boundary equilibrium (vmex.core.implicit)")
# hot_restart seeds each host VMEC solve from the previous boundary's converged
# state (few iterations as the boundary drifts); the loose adjoint tolerance
# matches the trust region's ~1e-3 gradient need -- together these make every
# warm objective evaluation a few seconds.  Trial boundaries that trip the VMEC
# initial-Jacobian guard are caught and penalized in ``scipy_fun`` below.
cfg = vmec_implicit.make_config(inp0, hot_restart=True,
                                adjoint_tol=1e-6, adjoint_maxiter=40)
params0 = vmec_implicit.params_from_input(inp0)

dof_modes = vmec_optimize._dof_modes(inp0, MAX_MODE)
NM = len(dof_modes)
NTOR = int(inp0.ntor)
dof_rows = np.asarray([n + NTOR for (_, n) in dof_modes])
dof_cols = np.asarray([m for (m, _) in dof_modes])
dof_names = vmec_optimize.boundary_dof_names(inp0, MAX_MODE)
dofs0 = jnp.asarray(vmec_optimize.pack_boundary(inp0, MAX_MODE))
if DOFS_INIT:
    dofs0 = jnp.asarray(np.load(DOFS_INIT)["dofs"])
    print(f"  resumed stage-B starting point from checkpoint: {DOFS_INIT}")
print(f"  QA starting point: aspect {aspect_shaped:.3f}, iota {iota_shaped:.3f} "
      f"(nfp={NFP}, ns={NS}, ftol={VMEC_FTOL:g})")
print(f"  boundary dofs:  {2 * NM} (RBC/ZBS with m,|n| <= {MAX_MODE}; RBC(0,0) fixed)")

qs_metric = vmec_optimize.QuasisymmetryRatioResidual(
    surfaces=QS_SURFACES, helicity_m=1, helicity_n=0
)

# Boozer output mode numbers (fixed by MBOZ/NBOZ/NFP; the booz_xform ordering)
_bm, _bn = [], []
for _m in range(MBOZ):
    for _n in range(0, NBOZ + 1) if _m == 0 else range(-NBOZ, NBOZ + 1):
        _bm.append(_m)
        _bn.append(_n * NFP)
BOOZ_XM = np.asarray(_bm)
BOOZ_XN = np.asarray(_bn)
# Quasi-axisymmetry in the Boozer spectrum: |B| = |B|(s, theta_B), so every
# n != 0 mode breaks the symmetry.  IDX_B00 locates the (0,0) normalization.
QS_BREAKING = jnp.asarray(BOOZ_XN != 0)
IDX_B00 = int(np.where((BOOZ_XM == 0) & (BOOZ_XN == 0))[0][0])

# ----------------------------------------------------------------------------
# 3) Kinetic operator template (built once; geometry replaced per evaluation)
# ----------------------------------------------------------------------------
print("Step 2: kinetic-operator template (canonical KineticOperator route)")
PSI_A_HAT = abs(float(inp0.phiedge)) / (2.0 * np.pi)
A_HAT = 1.0 / TARGET_ASPECT  # minor radius in RBar units for R0 ~ 1 m

KINETIC_TEMPLATE = f"""! Template generated by examples/optimize_QA_bootstrap.py
&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 1
  helicity_n = {NFP}
  psiAHat = {PSI_A_HAT:.10g}
  aHat = {A_HAT:.10g}
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = 1.0d+0 1.0d+0
  THats = 1.0d+0 1.0d+0
  dNHatdpsiHats = -1.5d+0 -1.5d+0
  dTHatdpsiHats = -3.0d+0 -3.0d+0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = {NU_N}
  Er = {ER}
  collisionOperator = 1
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
/
&resolutionParameters
  Ntheta = {KIN_NTHETA}
  Nzeta = {KIN_NZETA}
  Nxi = {KIN_NXI}
  NL = {KIN_NL}
  Nx = {KIN_NX}
  solverTolerance = {KINETIC_SOLVER_TOL}
/
&otherNumericalParameters
  xGridScheme = 5
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
"""
op_template = kinetic_operator_from_namelist(parse_sfincs_input_text(KINETIC_TEMPLATE))
S_KINETIC_ROW = NS // 2  # half-mesh radial row of the kinetic flux surface
S_KINETIC = (S_KINETIC_ROW - 0.5) / (NS - 1)  # normalized toroidal flux of that row

# The kinetic angular grids (theta/zeta nodes and quadrature weights) are
# fixed by the template resolution.
_kin_grids = make_grids(
    n_theta=op_template.n_theta, n_zeta=op_template.n_zeta, n_xi=op_template.n_xi,
    n_x=op_template.n_x, n_l=KIN_NL, n_periods=NFP, x_grid_scheme=5,
)
_KIN_THETA, _KIN_ZETA = _kin_grids.theta, _kin_grids.zeta
_KIN_THETA_W, _KIN_ZETA_W = _kin_grids.theta_weights, _kin_grids.zeta_weights
print(f"  species: ions + electrons, PAS collisions, nu_n={NU_N}, Er={ER}")
print(f"  grids: Ntheta={KIN_NTHETA} Nzeta={KIN_NZETA} Nxi={KIN_NXI} Nx={KIN_NX} "
      f"(matrix size {op_template.total_size})")
print(f"  kinetic flux surface: half-mesh row {S_KINETIC_ROW} of {NS} (s ~ {S_KINETIC:.3f})")


# ----------------------------------------------------------------------------
# 4) The differentiable physics chain (stage B), written out in this script
# ----------------------------------------------------------------------------
def kinetic_moments(booz, x0=None, recycle=None):
    """Solve the drift-kinetic equation on the Boozer surface; return moments."""
    bmnc_b = booz["bmnc_b"][0]
    ixm = np.asarray(booz["ixm_b"])
    ixn = np.asarray(booz["ixn_b"])  # includes the nfp factor
    geom = FluxSurfaceGeometry.from_fourier(
        theta=_KIN_THETA, zeta=_KIN_ZETA, bmnc=bmnc_b,
        m=jnp.asarray(ixm), n=jnp.asarray(ixn // NFP),
        n_periods=NFP, iota=booz["iota_b"][0], g_hat=booz["bvco_b"][0],
        i_hat=booz["buco_b"][0],
    )
    fsab2 = geom.fsab_hat2(theta_weights=_KIN_THETA_W, zeta_weights=_KIN_ZETA_W)
    op = dataclasses.replace(
        op_template,
        b_hat=geom.b_hat, db_hat_dtheta=geom.db_hat_dtheta, db_hat_dzeta=geom.db_hat_dzeta,
        d_hat=geom.d_hat, b_hat_sup_theta=geom.b_hat_sup_theta,
        b_hat_sup_zeta=geom.b_hat_sup_zeta, b_hat_sub_theta=geom.b_hat_sub_theta,
        b_hat_sub_zeta=geom.b_hat_sub_zeta, fsab_hat2=fsab2,
    )
    result = kinetic_solve(op, op.rhs(), method="gmres", tol=KINETIC_SOLVER_TOL,
                           differentiable=True, x0=x0, recycle=recycle)
    return profile_moments_from_operator(op, result.x), result


def _smooth_abs(x, eps=1e-8):
    return jnp.sqrt(x * x + eps * eps)


KINETIC_OBJECTIVES = {
    # squared normalized bootstrap current <j.B>/sqrt(<B^2>) -> drive to zero
    "bootstrap_jbs2": lambda mom: mom["FSABjHatOverRootFSAB2"] ** 2,
    # L1-style smooth sum of |radial particle flux| over species
    "particle_flux_l1": lambda mom: jnp.sum(_smooth_abs(mom["particleFlux_vm_psiHat"])),
    # squared radial heat fluxes summed over species
    "heat_flux_l2": lambda mom: jnp.sum(mom["heatFlux_vm_psiHat"] ** 2),
}


def objective(dofs, warm=None):
    """Total stage-B objective and diagnostics dict; differentiable in ``dofs``."""
    rbc = params0.rbc.at[dof_rows, dof_cols].set(dofs[:NM])
    zbs = params0.zbs.at[dof_rows, dof_cols].set(dofs[NM:])
    params = dataclasses.replace(params0, rbc=rbc, zbs=zbs)
    state = vmec_implicit.solve_implicit(params, cfg)  # custom-VJP equilibrium
    rt = vmec_implicit.runtime_from_params(params, cfg)

    aspect = vmec_optimize.aspect_ratio(state, rt)
    iota_mean = vmec_optimize.mean_iota(state, rt)
    qs_profile = qs_metric.total_state(state, rt)

    tabs = boozer_input_tables(state, rt, S_KINETIC_ROW)
    booz = booz_transform(
        rmnc=tabs["rmnc"][None, :], zmns=tabs["zmns"][None, :], lmns=tabs["lmns"][None, :],
        bmnc=tabs["bmnc"][None, :], bsubumnc=tabs["bsubumnc"][None, :],
        bsubvmnc=tabs["bsubvmnc"][None, :], iota=tabs["iota"][None],
        xm=tabs["xm"], xn=tabs["xn"], xm_nyq=tabs["xm"], xn_nyq=tabs["xn"],
        nfp=NFP, mboz=MBOZ, nboz=NBOZ, asym=False,
    )
    # Boozer-spectrum QA metric: energy fraction of the symmetry-breaking
    # (n != 0) modes of |B| on the kinetic surface, normalized to B00^2.
    bmnc_b = booz["bmnc_b"][0]
    qs_booz = jnp.sum(jnp.where(QS_BREAKING, bmnc_b, 0.0) ** 2) / bmnc_b[IDX_B00] ** 2

    mom, result = kinetic_moments(
        booz,
        x0=None if warm is None else warm.get("x0"),
        recycle=None if warm is None else warm.get("recycle"),
    )
    kinetic_term = KINETIC_OBJECTIVES[KINETIC_OBJECTIVE](mom)
    jbs = mom["FSABjHatOverRootFSAB2"]

    # Hard one-sided cap that HOLDS quasisymmetry at the Stage-A level: it is
    # exactly zero while the two-term ratio residual stays at/under
    # QS_HELD_TARGET and rises steeply once it tries to exceed it, so the
    # bootstrap search can spend at most the documented QA slack.
    qs_excess = jnp.maximum(qs_profile - QS_HELD_TARGET, 0.0)
    total = (W_ASPECT * (aspect - TARGET_ASPECT) ** 2
             + W_IOTA * jnp.maximum(IOTA_PEN_FLOOR - iota_mean, 0.0) ** 2
             + W_QS_BOOZ * qs_booz
             + W_QS_PROFILE * qs_profile
             + W_QS_HOLD * qs_excess ** 2
             + W_KINETIC * kinetic_term)
    sg = jax.lax.stop_gradient
    aux = {
        "aspect": aspect, "iota": iota_mean, "qs": qs_booz, "qs_profile": qs_profile,
        "qs_excess": qs_excess, "jbs": jbs, "kinetic_term": kinetic_term,
        "bmnc_b": sg(booz["bmnc_b"][0]),
        "booz_iota": sg(booz["iota_b"][0]),
        "booz_G": sg(booz["bvco_b"][0]),
        "booz_I": sg(booz["buco_b"][0]),
        "particle_flux": sg(mom["particleFlux_vm_psiHat"]),
        "heat_flux": sg(mom["heatFlux_vm_psiHat"]),
        "x_solution": sg(result.x),
        "recycle": (None if result.recycle is None
                    else tuple(sg(r) for r in result.recycle)),
        "kinetic_iterations": result.iterations,  # None under tracing
    }
    return total, aux


# ----------------------------------------------------------------------------
# 5) Initial evaluation + warm-start savings of the kinetic Krylov solve
# ----------------------------------------------------------------------------
print("Step 3: initial evaluation (cold kinetic solve)")
t0 = time.perf_counter()
J0, aux0 = objective(dofs0)
t_first = time.perf_counter() - t0
warm_state = {"x0": aux0["x_solution"], "recycle": aux0["recycle"]}
print(f"  objective J      = {float(J0):.6e}   ({t_first:.1f} s incl. JIT)")
print(f"  aspect ratio     = {float(aux0['aspect']):.4f} (target {TARGET_ASPECT})")
print(f"  mean iota        = {float(aux0['iota']):.4f} (require > {IOTA_MIN})")
print(f"  QS residual      = {float(aux0['qs']):.4e} (Boozer non-QA energy fraction)")
print(f"  QS ratio residual= {float(aux0['qs_profile']):.4e} ({len(QS_SURFACES)} surfaces)")
print(f"  <j.B>/sqrt(<B^2>)= {float(aux0['jbs']):.6e}")
print(f"  cold kinetic iterations = {aux0['kinetic_iterations']}")

print("Step 4: warm-start savings (x0 + GCROT recycle from the previous solve)")
t0 = time.perf_counter()
_, aux_warm = objective(dofs0, warm_state)
t_warm = time.perf_counter() - t0
it_cold = int(aux0["kinetic_iterations"])
it_warm = int(aux_warm["kinetic_iterations"])
print(f"  kinetic iterations: cold {it_cold} -> warm {it_warm} "
      f"({100.0 * (1.0 - it_warm / max(it_cold, 1)):.0f}% fewer)")
print(f"  wall time per objective evaluation: first {t_first:.1f} s -> warm {t_warm:.1f} s")

# ----------------------------------------------------------------------------
# 6) End-to-end gradient, checked against central finite differences
# ----------------------------------------------------------------------------
print("Step 5: jax.value_and_grad through equilibrium + Boozer + kinetic solve")
value_and_grad = jax.value_and_grad(objective, has_aux=True)
t0 = time.perf_counter()
(J_check, _), grad0 = value_and_grad(dofs0, warm_state)
t_grad = time.perf_counter() - t0
grad0_np = np.asarray(grad0)
print(f"  |grad| = {np.linalg.norm(grad0_np):.4e}   ({t_grad:.1f} s)")


def fd_gradient_check(dofs, grad_np, label):
    """Central-FD check of the dominant gradient component at ``dofs``."""
    k = int(np.argmax(np.abs(grad_np)))
    vp, _ = objective(dofs.at[k].add(FD_EPS), warm_state)
    vm, _ = objective(dofs.at[k].add(-FD_EPS), warm_state)
    fd = (float(vp) - float(vm)) / (2.0 * FD_EPS)
    rel = abs(grad_np[k] - fd) / max(abs(fd), 1e-300)
    print(f"  FD check ({label}) on dof {k} ({dof_names[k]}): "
          f"AD={grad_np[k]:.8e} FD={fd:.8e} rel={rel:.2e}")
    return {"dof": k, "name": dof_names[k], "ad": float(grad_np[k]),
            "fd": fd, "rel": rel}


fd_check = None
if RUN_FD_CHECK:
    fd_check = fd_gradient_check(dofs0, grad0_np, "starting point")
    print("  (the Boozer->kinetic segment alone is accurate to ~1e-6; the full")
    print("   chain is limited by the host equilibrium solver's ftol noise in FD)")
    if not (np.isfinite(fd_check["fd"]) and fd_check["rel"] < FD_GATE):
        raise SystemExit(f"end-to-end gradient check FAILED: rel {fd_check['rel']:.3e}")

# Arm the QA hold: cap the two-term QS ratio residual at QS_HELD_SLACK x its
# Stage-A (starting-point) value, so stage B reduces the bootstrap current at
# (essentially) fixed quasisymmetry.  Set here -- after the starting-point
# diagnostics above, which run on the smooth uncapped chain -- so the cap only
# shapes the optimization loop below.
QS_HELD_TARGET = QS_HELD_SLACK * float(aux0["qs_profile"])
print(f"  QA hold armed: cap two-term QS ratio residual at {QS_HELD_TARGET:.4e} "
      f"({QS_HELD_SLACK:g}x the Stage-A level {float(aux0['qs_profile']):.4e})")

# ----------------------------------------------------------------------------
# 7) Optimize stage B: scipy L-BFGS-B on the JAX value-and-gradient
# ----------------------------------------------------------------------------
print(f"Step 6: L-BFGS-B, {MAXITER} iterations, kinetic objective '{KINETIC_OBJECTIVE}'")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / f"{STEM}_progress.log"
CHECKPOINT_PATH = OUT_DIR / f"{STEM}_checkpoint.npz"
LOG_PATH.write_text(
    "# eval  objective      <j.B>/sqrt(<B^2>)  qs_booz     qs_profile  "
    "iota     aspect   wall_s\n"
)
history = []
_eval_index = [0]
_best = [float("inf"), np.asarray(dofs0, dtype=float)]


def scipy_fun(x):
    """value+gradient wrapper with kinetic warm starts across evaluations."""
    t_start = time.perf_counter()
    try:
        (value, aux), grad = value_and_grad(jnp.asarray(x), warm_state)
    except Exception as exc:  # zero-crash policy: penalize failed equilibria
        print(f"  eval (failed trial boundary, penalized): {type(exc).__name__}")
        return PENALTY_VALUE, np.zeros_like(np.asarray(x, dtype=float))
    # reuse this evaluation's kinetic solution/recycle pair for the next one
    warm_state["x0"] = aux["x_solution"]
    warm_state["recycle"] = aux["recycle"]
    _eval_index[0] += 1
    record = {
        "eval": _eval_index[0],
        "objective": float(value),
        "aspect": float(aux["aspect"]),
        "iota": float(aux["iota"]),
        "qs": float(aux["qs"]),
        "qs_profile": float(aux["qs_profile"]),
        "jbs": float(aux["jbs"]),
        "wall_s": time.perf_counter() - t_start,
    }
    history.append(record)
    print(f"  eval {record['eval']:3d}: J={record['objective']:.6e} "
          f"qs={record['qs']:.3e} <j.B>={record['jbs']:+.4e} "
          f"aspect={record['aspect']:.3f} iota={record['iota']:.4f} "
          f"({record['wall_s']:.1f} s)")
    with LOG_PATH.open("a") as fh:
        fh.write(f"{record['eval']:6d}  {record['objective']:.6e}  "
                 f"{record['jbs']:+.10e}  {record['qs']:.4e}  "
                 f"{record['qs_profile']:.4e}  {record['iota']:.5f}  "
                 f"{record['aspect']:.4f}  {record['wall_s']:.1f}\n")
    if record["objective"] < _best[0]:
        _best[0] = record["objective"]
        _best[1] = np.asarray(x, dtype=float).copy()
        np.savez(CHECKPOINT_PATH, dofs=_best[1], objective=_best[0],
                 eval_index=record["eval"])
    return float(value), np.asarray(grad, dtype=float)


bounds = [(float(v) - BOUND_RADIUS, float(v) + BOUND_RADIUS) for v in np.asarray(dofs0)]
opt = scipy.optimize.minimize(
    scipy_fun, np.asarray(dofs0, dtype=float), jac=True, method="L-BFGS-B",
    bounds=bounds, options={"maxiter": MAXITER, "maxcor": 20},
)
dofs_final = jnp.asarray(opt.x)
J_final, aux_final = objective(dofs_final, warm_state)
print(f"  optimizer stop: {opt.message}")

fd_check_final = None
if RUN_FD_CHECK:
    (_, _), grad_final = value_and_grad(dofs_final, warm_state)
    fd_check_final = fd_gradient_check(dofs_final, np.asarray(grad_final), "final point")

# ----------------------------------------------------------------------------
# 8) Final results, saved outputs (optimized input + wout), read-back, plot
# ----------------------------------------------------------------------------
print("Step 7: final results (precise QA vs precise QA held + bootstrap)")
jbs0, jbs1 = float(aux0["jbs"]), float(aux_final["jbs"])
boot_x, boot_y = abs(jbs0), abs(jbs1)
boot_factor = boot_x / max(boot_y, 1e-300)
r_a, r_b = float(aux0["qs_profile"]), float(aux_final["qs_profile"])
print(f"  QA shaping:      QS ratio {qs_seed:.3e} -> {qs_shaped:.3e} "
      f"(circular torus -> QA), iota {iota_seed:.3f} -> {iota_shaped:.3f}")
print(f"  objective        {float(J0):.6e} -> {float(J_final):.6e}")
print("  config 1 = precise QA (no bootstrap opt);  "
      "config 2 = precise QA held + bootstrap opt")
print(f"  X = <j.B>/sqrt(<B^2>) config 1 = {jbs0:+.6e}")
print(f"  Y = <j.B>/sqrt(<B^2>) config 2 = {jbs1:+.6e}")
print(f"  bootstrap decrease at held QA:  |X| {boot_x:.6e} -> |Y| {boot_y:.6e}  "
      f"({boot_factor:.2f}x lower)")
print(f"  two-term QS ratio residual (HELD): {r_a:.4e} (config 1) -> {r_b:.4e} (config 2)"
      f"   [cap {QS_HELD_TARGET:.4e} = {QS_HELD_SLACK:g}x config 1]")
print(f"  Boozer non-QA fraction: {float(aux0['qs']):.4e} -> {float(aux_final['qs']):.4e}")
print(f"  aspect ratio     {float(aux0['aspect']):.4f} -> {float(aux_final['aspect']):.4f} "
      f"(target {TARGET_ASPECT})")
print(f"  mean iota        {float(aux0['iota']):.4f} -> {float(aux_final['iota']):.4f} "
      f"(require > {IOTA_MIN})")
objective_decreased = float(J_final) < float(J0)
print(f"  objective decreased: {objective_decreased}")

inp_final = vmec_optimize.unpack_boundary(inp0, np.asarray(dofs_final, dtype=float), MAX_MODE)
input_path = inp_final.to_indata(OUT_DIR / f"input.{STEM}_optimized")
np.savez(OUT_DIR / f"{STEM}_dofs_final.npz", dofs=np.asarray(dofs_final, dtype=float))

wout_paths = {}
wouts = {}
for tag, inp_tag in (("initial", inp0), ("final", inp_final)):
    res = vmec_solver.solve(inp_tag, cfg.resolution, ftol=cfg.ftol,
                            max_iterations=cfg.max_iterations, mode="cli")
    wout = wout_from_state(inp=inp_tag, state=res.state, fsqr=res.fsqr,
                           fsqz=res.fsqz, fsql=res.fsql)
    wouts[tag] = wout
    wout_paths[tag] = write_wout(OUT_DIR / f"wout_{STEM}_{tag}.nc", wout)

history_path = OUT_DIR / f"{STEM}_history.json"
history_path.write_text(json.dumps({
    "history": history,
    "shaping": {"qs_seed": qs_seed, "qs_shaped": qs_shaped,
                "iota_seed": iota_seed, "iota_shaped": iota_shaped,
                "aspect_shaped": aspect_shaped, "wall_s": t_shape},
    "initial": {k: float(aux0[k]) for k in ("aspect", "iota", "qs", "qs_profile", "jbs")},
    "final": {k: float(aux_final[k]) for k in ("aspect", "iota", "qs", "qs_profile", "jbs")},
    "objective_initial": float(J0), "objective_final": float(J_final),
    "warm_start": {"iterations_cold": it_cold, "iterations_warm": it_warm,
                   "seconds_first": t_first, "seconds_warm": t_warm},
    "fd_check_initial": fd_check, "fd_check_final": fd_check_final,
    "kinetic_objective": KINETIC_OBJECTIVE,
    "targets": {"aspect": TARGET_ASPECT, "iota_min": IOTA_MIN,
                "qs_held_target": float(QS_HELD_TARGET),
                "qs_held_slack": float(QS_HELD_SLACK)},
    "bootstrap_at_held_qa": {"X_config1": boot_x, "Y_config2": boot_y,
                             "factor": boot_factor,
                             "qs_ratio_config1": r_a, "qs_ratio_config2": r_b},
    "dof_names": dof_names,
}, indent=2) + "\n")
read_back = json.loads(history_path.read_text())
print(f"  read back from json: objective_final = {read_back['objective_final']:.6e}")


def _boundary_surface_bmag(wout, ntheta=96, nzeta=256):
    """(X, Y, Z, |B|) on the outermost flux surface over the full torus."""
    xm = np.asarray(wout.xm, dtype=float)
    xn = np.asarray(wout.xn, dtype=float)
    xm_nyq = np.asarray(wout.xm_nyq, dtype=float)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=float)
    rmnc = np.asarray(wout.rmnc)[-1]  # boundary row, full mesh
    zmns = np.asarray(wout.zmns)[-1]
    bmnc = np.asarray(wout.bmnc)[-1]  # outermost half-mesh row (s = 1 - h/2)
    th = np.linspace(0.0, 2.0 * np.pi, ntheta)
    ze = np.linspace(0.0, 2.0 * np.pi, nzeta)
    tg, zg = np.meshgrid(th, ze, indexing="ij")
    ang = xm[None, None, :] * tg[:, :, None] - xn[None, None, :] * zg[:, :, None]
    rr = np.einsum("m,tzm->tz", rmnc, np.cos(ang))
    zz = np.einsum("m,tzm->tz", zmns, np.sin(ang))
    ang_nyq = (xm_nyq[None, None, :] * tg[:, :, None]
               - xn_nyq[None, None, :] * zg[:, :, None])
    bb = np.einsum("m,tzm->tz", bmnc, np.cos(ang_nyq))
    return rr * np.cos(zg), rr * np.sin(zg), zz, bb


fig = plt.figure(figsize=(12.8, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.3), hspace=0.55, wspace=0.3,
                      left=0.11, right=0.98, top=0.91, bottom=0.25)

cfg_colors = ["#8c8c8c", "#d62728"]  # config 1 (precise QA) / config 2 (held + boot)
cfg_labels = [
    f"precise QA\n(no bootstrap opt)\naspect {float(aux0['aspect']):.2f}, "
    f"iota {float(aux0['iota']):.3f}",
    f"precise QA held\n+ bootstrap opt\naspect {float(aux_final['aspect']):.2f}, "
    f"iota {float(aux_final['iota']):.3f}",
]

# Panel A: bootstrap current of the two precise-QA configs (X vs Y).
axA = fig.add_subplot(gs[0, 0])
boot_vals = [boot_x, boot_y]
for i, (v, c) in enumerate(zip(boot_vals, cfg_colors)):
    axA.bar(i, v, width=0.62, color=c)
    axA.annotate(f"{v:.2e}", xy=(i, v), xytext=(0, 3), textcoords="offset points",
                 ha="center", fontsize=9)
axA.set_ylabel(r"$|\langle j\!\cdot\!B\rangle|/\sqrt{\langle B^2\rangle}$")
axA.set_title(f"bootstrap current  ({boot_factor:.2f}x lower at held QA)", fontsize=10.5)
axA.set_xticks([0, 1])
axA.set_xticklabels(["", ""])
axA.set_xlim(-0.6, 1.6)
axA.set_ylim(0, max(boot_vals) * 1.3)

# Panel B: two-term QS ratio residual -- HELD near the Stage-A level.
axB = fig.add_subplot(gs[1, 0])
qs_vals = [r_a, r_b]
for i, (v, c) in enumerate(zip(qs_vals, cfg_colors)):
    axB.bar(i, v, width=0.62, color=c)
    axB.annotate(f"{v:.2e}", xy=(i, v), xytext=(0, 3), textcoords="offset points",
                 ha="center", fontsize=9)
axB.axhline(QS_HELD_TARGET, ls="--", lw=0.9, color="k")
axB.annotate(f"hold cap = {QS_HELD_SLACK:g}x", xy=(1.58, QS_HELD_TARGET), xytext=(0, 2),
             textcoords="offset points", ha="right", va="bottom", fontsize=8)
axB.set_ylabel("two-term QS\nratio residual")
axB.set_title("quasisymmetry held (both precise QA)", fontsize=10.5)
axB.set_xticks([0, 1])
axB.set_xticklabels(cfg_labels, fontsize=8)
axB.set_xlim(-0.6, 1.6)
axB.set_ylim(0, max(max(qs_vals), float(QS_HELD_TARGET)) * 1.35)

X, Y, Z, B = _boundary_surface_bmag(wouts["final"])
norm = matplotlib.colors.Normalize(vmin=B.min(), vmax=B.max())
ax3d = fig.add_subplot(gs[:, 1], projection="3d")
ax3d.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(norm(B)),
                  rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
ax3d.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)), zoom=1.55)
ax3d.set_axis_off()
ax3d.view_init(elev=32, azim=-65)
ax3d.set_title("config 2 boundary, |B| (T)", fontsize=11, pad=0)
mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
cbar = fig.colorbar(mappable, ax=ax3d, shrink=0.62, pad=0.0, fraction=0.04)
cbar.ax.tick_params(labelsize=8)

plot_path = OUT_DIR / f"{STEM}.png"
fig.savefig(plot_path, dpi=120)
plt.close(fig)

print(f"  Saved plot: {plot_path}")
print(f"  Wrote output files: {input_path.name}, "
      f"{wout_paths['initial'].name}, {wout_paths['final'].name}, {history_path.name}, "
      f"{LOG_PATH.name}, {CHECKPOINT_PATH.name}")
print("Done: examples/optimize_QA_bootstrap.py")
