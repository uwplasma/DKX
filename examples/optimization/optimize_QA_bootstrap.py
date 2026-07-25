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
    needs ~1e-3 gradients) -- which is what keeps a momentum-conserving
    collision operator affordable inside the optimization loop; and

  - verifying the end-to-end gradient against central finite differences at
    the starting point AND at the optimized end point.

Physics: quasi-axisymmetry makes the guiding-centre drifts tokamak-like, so a
QA field carries a substantial bootstrap current -- lowering it at fixed
quasisymmetry, aspect ratio and iota is a genuine Pareto trade, not a free
lunch (holding QA fixed makes the achievable bootstrap decrease honestly
smaller than a search that is free to spend quasisymmetry).  The kinetic
configuration is the classic "full trajectories" setup -- two species (ions and
electrons) with a finite radial electric field carried by the full-trajectory
terms (includeXDotTerm and includeElectricFieldTermInXiDot on), which routes to
the tier-2 GCROT solver where warm starts and recycling matter -- at
reactor-core conditions, and with the *momentum-conserving* full linearized
Fokker-Planck collision operator, which the bootstrap current requires because
<j.B> is itself a parallel-momentum moment [Landreman, Smith, Mollen &
Helander, Phys. Plasmas 21, 042503 (2014); Helander & Sigmar, Collisional
Transport in Magnetized Plasmas, CUP (2002); Landreman & Paul, PRL 128, 035001
(2022)].  The kinetic surface sits in the banana regime, where the bootstrap
current is set by the field geometry the optimizer is allowed to change rather
than by collisions; the script prints that classification (dkx.validity) next to
the solved moments.

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

Expected runtime (measured on a laptop CPU, default resolution): stage A ~190 s,
dominated by the one-time implicit-Jacobian XLA compile per continuation stage
(warm forward solves ~1 s); stage B by the host VMEC solve plus the kinetic
value_and_grad -- the first objective evaluation ~31 s including compilation,
warm ones ~8 s, and one value_and_grad ~30 s.  The momentum-conserving
collision operator is most of that kinetic cost (a pitch-angle operator needs a
fraction of it but gets the bootstrap current wrong by tens of percent).  The
whole example runs in well under an hour.
Progress is appended to a per-evaluation log file and the best point is
checkpointed after every evaluation, so a long run is inspectable while it goes
and resumable (DKX_QA_DOFS_INIT).  With DKX_CI=1 everything
shrinks to a couple of minutes.

Requires the optional companions of this example (not needed by dkx
itself):  pip install -e /path/to/vmex /path/to/booz_xform_jax

Run:
  python examples/optimization/optimize_QA_bootstrap.py
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

from dkx.bounce_averaged import effective_ripple
from dkx.collisions import nu_d_hat_pitch_angle_scattering_v3
from dkx.constants import RadialCoordinates
from dkx.drift_kinetic import kinetic_operator_from_namelist
from dkx.inputs import parse_sfincs_input_text
from dkx.magnetic_geometry import FluxSurfaceGeometry
from dkx.phase_space import make_grids
from dkx.run import profile_moments_from_operator
from dkx.solve import solve as kinetic_solve
from dkx.validity import local_validity_report

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
# Trial budget per stage-A stage.  The CI budget has to be large enough to break
# the circular seed's iota saddle: a surface left at iota ~ 0 has no rotational
# transform to hold the parallel dynamics, and the drift-kinetic operator there
# is nearly singular (the Krylov solve stalls before the requested tolerance).
# Ten trials reach iota ~ 0.42 and cost no more wall time than three did.
SHAPE_MAX_NFEV = 10 if CI else 60
SHAPE_FTOL = 1e-3 if CI else 1e-5

# Boozer transform resolution for the kinetic flux surface.
MBOZ, NBOZ = (2, 2) if CI else (4, 4)

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

# ----------------------------------------------------------------------------
# Kinetic solve: resolution, which flux surface, and the conditions on it
# ----------------------------------------------------------------------------
# Two species (ions + electrons) with a finite radial electric field carried by
# the full-trajectory terms (includeXDotTerm / includeElectricFieldTermInXiDot),
# which routes the system to the tier-2 GCROT solver where warm starts and
# recycling matter; it is also nonsingular and exactly implicit-differentiable.
KIN_NTHETA, KIN_NZETA, KIN_NXI, KIN_NL, KIN_NX = (9, 7, 8, 4, 4) if CI else (13, 11, 24, 4, 6)
KINETIC_SOLVER_TOL = 1e-9
S_KINETIC_ROW = NS // 2  # half-mesh radial row carrying the kinetic flux surface
S_KINETIC = (S_KINETIC_ROW - 0.5) / (NS - 1)  # normalized toroidal flux of that row

# COLLISION OPERATOR.  <j.B> is a parallel-momentum moment of the distribution
# function, so the operator has to conserve parallel momentum or the bootstrap
# current is systematically wrong.  Pitch-angle scattering (collisionOperator=1)
# conserves particles but not momentum: it drops the field-particle
# back-reaction through which each species returns the momentum it takes from
# the others.  The full linearized Fokker-Planck operator (collisionOperator=0)
# carries that term and conserves momentum exactly, which is what
# bootstrap-current calculations require [Helander & Sigmar, Collisional
# Transport in Magnetized Plasmas, CUP (2002), ch. 8 and 11; Landreman, Smith,
# Mollen & Helander, Phys. Plasmas 21, 042503 (2014)].  Measured on the Stage-A
# QA surface at the conditions and resolution below, pitch-angle scattering
# returns <j.B>/sqrt(<B^2>) = 2.383e-1 against 1.626e-1 for the full operator --
# a 47% overestimate that is purely the missing momentum restoration (in a
# collisional deck the same comparison is a factor 2.8).  The momentum- AND
# energy-conserving improved Sugama model operator (collisionOperator=3; Sugama
# et al., Phys. Plasmas 26, 102108 (2019)) is the cheaper fallback -- 25% fewer
# Krylov iterations here, and within 18% of the full operator at this
# resolution -- but the full operator is affordable, so the flagship uses it.
# Nxi = 24 (against 16) is what the banana regime costs: the pitch-angle
# resolution the collisional deck could get away with leaves <j.B> 8% low, while
# 24 is within 1% of Nxi = 32.
COLLISION_OPERATOR = 0

# NORMALIZATION.  Delta = mBar vBar/(e BBar RBar) fixes the reference set
# (RBar = 1 m, BBar = 1 T, TBar = 1 keV, mBar = m_proton, nBar = 1e20 m^-3), and
# nu_n = nuBar RBar/vBar is the collisionality AT those reference values -- a
# pure normalization constant, not a plasma property: 0.00831565 with a Coulomb
# logarithm of 17.  The actual plasma enters through nHats/THats below.  Raising
# nu_n instead of the densities is what puts a case in the wrong regime: 12x
# this value gives nu_star = 2.2e-1, six to ten times the banana boundary
# eps_t^1.5, which is the plateau regime -- there the bootstrap current is set
# by collisions rather than by the geometry the optimizer may change.  Step 3b
# below prints where the surface actually lands.
DELTA = 4.5694e-3
NU_N = 0.00831565
ER = 1.0  # radial electric field in kV/m (phiBar/RBar); ion-root scale

# REACTOR SCALE.  Stage A shapes a unit-scale equilibrium (R0 ~ 1 m, |B| ~ 1 T).
# Aspect ratio, iota and quasisymmetry are scale invariant, so the shaping stage
# never needs a machine size -- but neoclassical transport is not scale
# invariant: both the collisionality and the orbit width depend on size and
# field strength.  The kinetic surface is therefore rigidly rescaled to reactor
# dimensions before the drift-kinetic solve.  A rigid scale (lengths x L, field
# x S) maps a Boozer surface exactly: |B| -> S |B|, (G, I) -> L S (G, I),
# psiAHat -> L^2 S psiAHat, iota unchanged.  Reactor-core density and
# temperature on the unit-scale torus would instead mean beta of order 1 and
# banana orbits wider than the temperature scale length (measured delta_FOW =
# w_b/L_T = 1.4), where no radially-local kinetic model applies at all.
A_MINOR = 1.70442623  # m, reactor minor radius of the reference QA case
B_AXIS = 5.9021049  # T, |B_00| of the same reference case
A_HAT_UNIT = 1.0 / TARGET_ASPECT  # unit-scale minor radius (R0 ~ 1 m units)
R_SCALE = A_MINOR / A_HAT_UNIT  # length scale factor of the rigid rescale

# PROFILES.  Reactor-core density and temperature of the reference
# bootstrap-current study, evaluated at the kinetic surface:
#     n(s) = 4.13 (1 - s^5) x 1e20 m^-3,     T(s) = 12 (1 - s) keV,
# with s the normalized toroidal flux.  nBar = 1e20 m^-3 and TBar = 1 keV, so
# nHat and THat ARE those numbers.  rHat = A_MINOR sqrt(s) is the flux-surface
# radius, hence d/drHat = (2 sqrt(s)/A_MINOR) d/ds for the gradient drives
# [Landreman, Buller & Drevlak, arXiv:2205.02914].
N_AXIS, T_AXIS = 4.13, 12.0  # 1e20 m^-3 and keV on axis
_DS_DRHAT = 2.0 * np.sqrt(S_KINETIC) / A_MINOR
NHAT = N_AXIS * (1.0 - S_KINETIC**5)  # 1e20 m^-3 on the kinetic surface
THAT = T_AXIS * (1.0 - S_KINETIC)  # keV on the kinetic surface
DNHAT_DRHAT = -5.0 * N_AXIS * S_KINETIC**4 * _DS_DRHAT
DTHAT_DRHAT = -T_AXIS * _DS_DRHAT

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
# the host equilibrium solver's ftol termination noise, which a finite
# difference divides by eps.  Measured eps sweep on the dominant dof at the
# default resolution, with the autodiff value fixed at 2.18005e+06:
#     3e-5 -> 3.40e-2,  1e-4 -> 2.93e-2,  3e-4 -> 1.72e-2,  1e-3 -> 1.27e-2,
# i.e. the FD value climbs monotonically toward the stable autodiff value as
# the step grows out of the noise, with no sign of truncation error yet.  So
# FD_EPS sits at the top of that sweep and the gate accommodates the remaining
# noise floor -- the comparison is limited by finite differences, not by the
# autodiff chain.
FD_EPS = 1e-5 if CI else 1e-3
FD_GATE = 5e-2 if CI else 2.5e-2

# Optional resume: point DKX_QA_DOFS_INIT at a checkpoint .npz written
# by a previous (interrupted) run to start stage B from its best point.
DOFS_INIT = os.environ.get("DKX_QA_DOFS_INIT", "")

OUT_DIR = Path(os.environ.get("DKX_QA_OUT_DIR", str(Path(__file__).parent / "output")))
STEM = "optimize_QA_bootstrap"

# ----------------------------------------------------------------------------
# 1) Stage A: circular torus -> precise QA via vmex.optimize.least_squares
# ----------------------------------------------------------------------------
print("=== examples/optimization/optimize_QA_bootstrap.py ===")
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
# The rigid rescale from the unit-scale equilibrium to the reactor dimensions
# declared at the top: B_UNIT is the flux-consistent field of the unit-scale
# surface (psiAHat = B a^2/2), so the rescaled surface carries |B| ~ B_AXIS at
# minor radius A_MINOR, and its psiAHat is B_AXIS A_MINOR^2/2.
PSI_A_HAT_UNIT = abs(float(inp0.phiedge)) / (2.0 * np.pi)
B_UNIT = 2.0 * PSI_A_HAT_UNIT / A_HAT_UNIT**2
B_SCALE = B_AXIS / B_UNIT  # field scale factor of the rigid rescale
PSI_A_HAT = PSI_A_HAT_UNIT * R_SCALE**2 * B_SCALE

KINETIC_TEMPLATE = f"""! Template generated by examples/optimization/optimize_QA_bootstrap.py
&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 1
  helicity_n = {NFP}
  psiAHat = {PSI_A_HAT:.10g}
  aHat = {A_MINOR:.10g}
  inputRadialCoordinate = 1
  psiN_wish = {S_KINETIC:.10g}
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = {NHAT:.10g} {NHAT:.10g}
  THats = {THAT:.10g} {THAT:.10g}
  dNHatdrHats = {DNHAT_DRHAT:.10g} {DNHAT_DRHAT:.10g}
  dTHatdrHats = {DTHAT_DRHAT:.10g} {DTHAT_DRHAT:.10g}
/
&physicsParameters
  Delta = {DELTA:.10g}
  alpha = 1.0d+0
  nu_n = {NU_N}
  Er = {ER}
  collisionOperator = {COLLISION_OPERATOR}
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

# The kinetic angular grids (theta/zeta nodes and quadrature weights) are
# fixed by the template resolution.
_kin_grids = make_grids(
    n_theta=op_template.n_theta, n_zeta=op_template.n_zeta, n_xi=op_template.n_xi,
    n_x=op_template.n_x, n_l=KIN_NL, n_periods=NFP, x_grid_scheme=5,
)
_KIN_THETA, _KIN_ZETA = _kin_grids.theta, _kin_grids.zeta
_KIN_THETA_W, _KIN_ZETA_W = _kin_grids.theta_weights, _kin_grids.zeta_weights
print(f"  species: ions + electrons, full linearized Fokker-Planck collisions "
      f"(collisionOperator={COLLISION_OPERATOR}, momentum-conserving), "
      f"nu_n={NU_N:g}, Er={ER:g} kV/m")
print(f"  reactor scale: minor radius {A_MINOR:.4f} m, |B| ~ {B_AXIS:.3f} T "
      f"(rigid rescale x{R_SCALE:.3f} in length, x{B_SCALE:.3f} in field)")
print(f"  profiles at s = {S_KINETIC:.4f}: n = {NHAT:.4f}e20 m^-3, T = {THAT:.4f} keV, "
      f"dn/drHat = {DNHAT_DRHAT:.5f}, dT/drHat = {DTHAT_DRHAT:.4f}")
print(f"  grids: Ntheta={KIN_NTHETA} Nzeta={KIN_NZETA} Nxi={KIN_NXI} Nx={KIN_NX} "
      f"(matrix size {op_template.total_size})")
print(f"  kinetic flux surface: half-mesh row {S_KINETIC_ROW} of {NS} (s ~ {S_KINETIC:.3f})")


# ----------------------------------------------------------------------------
# 4) The differentiable physics chain (stage B), written out in this script
# ----------------------------------------------------------------------------
def reactor_surface_geometry(booz):
    """Boozer surface of the equilibrium, rigidly rescaled to reactor size/field.

    Lengths x R_SCALE and field x B_SCALE: ``|B| -> B_SCALE |B|`` and
    ``(G, I) -> R_SCALE B_SCALE (G, I)`` (``iota`` is scale invariant), the
    exact image of the unit-scale surface at reactor dimensions.  Traceable in
    the spectrum, so gradients flow through the rescale unchanged.
    """
    ixm = np.asarray(booz["ixm_b"])
    ixn = np.asarray(booz["ixn_b"])  # includes the nfp factor
    return FluxSurfaceGeometry.from_fourier(
        theta=_KIN_THETA, zeta=_KIN_ZETA, bmnc=booz["bmnc_b"][0] * B_SCALE,
        m=jnp.asarray(ixm), n=jnp.asarray(ixn // NFP),
        n_periods=NFP, iota=booz["iota_b"][0],
        g_hat=booz["bvco_b"][0] * (B_SCALE * R_SCALE),
        i_hat=booz["buco_b"][0] * (B_SCALE * R_SCALE),
    )


def kinetic_moments(booz, x0=None, recycle=None):
    """Solve the drift-kinetic equation on the Boozer surface; return moments."""
    geom = reactor_surface_geometry(booz)
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


# ----------------------------------------------------------------------------
# 5b) Which transport regime the kinetic surface is in (dkx.validity)
# ----------------------------------------------------------------------------
def regime_report(aux):
    """Collisionality regime + radial-locality verdict for the kinetic surface.

    The bootstrap current is only 'geometry dominated' -- the premise of
    optimizing it by reshaping the boundary -- when the surface sits in the
    long-mean-free-path (banana) regime, ``nu_star < eps_t^{3/2}``.  Higher
    collisionality moves it into the plateau / Pfirsch-Schlueter regimes where
    collisions, not the field geometry, set the parallel current.
    """
    booz = {"bmnc_b": jnp.asarray(aux["bmnc_b"])[None, :],
            "ixm_b": BOOZ_XM, "ixn_b": BOOZ_XN,
            "iota_b": jnp.asarray(aux["booz_iota"])[None],
            "bvco_b": jnp.asarray(aux["booz_G"])[None],
            "buco_b": jnp.asarray(aux["booz_I"])[None]}
    geom = reactor_surface_geometry(booz)
    b00 = float(aux["bmnc_b"][IDX_B00]) * B_SCALE
    g_hat = float(aux["booz_G"]) * B_SCALE * R_SCALE
    i_hat = float(aux["booz_I"]) * B_SCALE * R_SCALE
    iota = float(aux["booz_iota"])
    r_major = (g_hat + iota * i_hat) / b00  # (G + iota I)/B0, the connection length
    r_hat = A_MINOR * np.sqrt(S_KINETIC)
    eps_t = r_hat / r_major  # inverse aspect ratio of the surface
    # nu_star = R0 nu_D/(iota v) for the thermal ion (x = 1) at these profiles
    nu_d = float(nu_d_hat_pitch_angle_scattering_v3(
        x=jnp.asarray([1.0]), z_s=jnp.asarray([1.0, -1.0]),
        m_hats=jnp.asarray([1.0, 5.446170214e-4]),
        n_hats=jnp.asarray([NHAT, NHAT]), t_hats=jnp.asarray([THAT, THAT]))[0, 0])
    nu_star = r_major * NU_N * nu_d / (abs(iota) * np.sqrt(THAT))
    radial = RadialCoordinates(psi_a_hat=PSI_A_HAT, a_hat=A_MINOR,
                               r_n=float(np.sqrt(S_KINETIC)))
    e_star = (radial.d_dr_hat_to_d_dpsi_hat * (-ER)) * DELTA * g_hat / (2.0 * iota * b00)
    eps_eff = float(effective_ripple(geom, r_eff=r_hat))
    l_grad = min(NHAT / abs(DNHAT_DRHAT), THAT / abs(DTHAT_DRHAT))  # min(L_n, L_T)
    return local_validity_report(
        nu_star=nu_star, e_star=e_star, delta=DELTA, g_hat=g_hat, iota=iota,
        b0_over_bbar=b00, eps_t=eps_t, grad_scale_length_hat=l_grad,
        t_hat=THAT, epsilon_eff=eps_eff), eps_eff, b00, r_major


print("Step 3b: transport regime of the kinetic surface (dkx.validity)")
validity, EPS_EFF, B00_REACTOR, R_MAJOR = regime_report(aux0)
BANANA_BOUNDARY = validity.eps_t**1.5
print(f"  surface: |B00| = {B00_REACTOR:.3f} T, R0 = {R_MAJOR:.3f} m, "
      f"eps_t = {validity.eps_t:.4f}, effective ripple = {EPS_EFF:.2e}")
print(f"  nu_star = {validity.nu_star:.4e} vs banana boundary eps_t^1.5 = "
      f"{BANANA_BOUNDARY:.4e}  ({validity.nu_star / BANANA_BOUNDARY:.2f}x)")
print(f"  regime = {validity.regime.value}  (E x B precession k_ExB = "
      f"{validity.k_exb:.3f}; collisions de-trap first while that stays below 1)")
print(f"  radial locality: delta_FOW = w_b/L = {validity.delta_fow:.4f} "
      f"({validity.radial_locality_flag.value}); overall {validity.overall_flag.value}")

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
    "kinetic_conditions": {
        "collision_operator": COLLISION_OPERATOR, "nu_n": NU_N, "Delta": DELTA,
        "Er_kV_per_m": ER, "s": S_KINETIC,
        "nHat_1e20_per_m3": NHAT, "THat_keV": THAT,
        "dNHatdrHat": DNHAT_DRHAT, "dTHatdrHat": DTHAT_DRHAT,
        "a_minor_m": A_MINOR, "B_axis_T": B_AXIS,
        "length_scale": R_SCALE, "field_scale": B_SCALE,
        "regime": validity.regime.value, "nu_star": validity.nu_star,
        "banana_boundary": BANANA_BOUNDARY, "epsilon_eff": EPS_EFF,
        "k_ExB": validity.k_exb, "delta_FOW": validity.delta_fow,
        "radial_locality": validity.radial_locality_flag.value,
    },
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
B = B * B_SCALE  # unit-scale equilibrium shown at the reactor field strength
norm = matplotlib.colors.Normalize(vmin=B.min(), vmax=B.max())
ax3d = fig.add_subplot(gs[:, 1], projection="3d")
ax3d.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(norm(B)),
                  rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
ax3d.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)), zoom=1.55)
ax3d.set_axis_off()
ax3d.view_init(elev=32, azim=-65)
ax3d.set_title("config 2 boundary, |B| (T) at reactor scale", fontsize=11, pad=0)
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
print("Done: examples/optimization/optimize_QA_bootstrap.py")
