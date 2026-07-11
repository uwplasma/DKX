"""Flagship optimization: a quasi-axisymmetric stellarator with low bootstrap current.

What this example teaches:
  - a full gradient-based stellarator optimization loop where the objective is
    a plain JAX function of the plasma-boundary Fourier coefficients, and the
    bootstrap current <j.B> comes from the canonical sfincs_jax kinetic solve
    (KineticOperator -> solve(differentiable=True) -> rhsmode1_moments),
  - the differentiable route between the codes:
        boundary dofs -> vmec_jax.core.implicit.solve_implicit (fixed-boundary
        MHD equilibrium with an implicit-adjoint custom VJP)
        -> traceable single-surface VMEC spectral tables
           (vmec_jax.core.boozer_tables.boozer_input_tables; validated
           against the host wout tables in tests/test_example_qa_bootstrap.py)
        -> booz_xform_jax (differentiable Boozer transform, |B| spectrum)
        -> FluxSurfaceGeometry.from_fourier (geometryScheme-13 pure-JAX path)
        -> KineticOperator -> tier-2 GCROT solve with implicit differentiation
        -> FSABjHat,
    so jax.value_and_grad returns the gradient of the *whole* physics chain,
  - warm-starting the kinetic Krylov solve across optimizer evaluations with
    the previous solution (x0) and the GCROT recycle pair, and
  - verifying the end-to-end gradient against central finite differences.

Physics: the starting point is the Landreman & Paul (2021) precise QA at low
resolution.  The field is already strongly quasisymmetric; the optimizer's job
is to reduce the bootstrap current <j.B>/sqrt(<B^2>) on a mid-radius surface
while penalty terms hold aspect ratio, mean iota and the two-term
quasisymmetry residual [Landreman & Paul, PRL 128, 035001 (2022)].  The
kinetic configuration is the classic SFINCS "full trajectories" setup:
two-species pitch-angle-scattering collisions with a finite radial electric
field (includeXDotTerm and includeElectricFieldTermInXiDot on), which routes
to the tier-2 GCROT solver where warm starts and recycling matter [Landreman,
Smith, Mollen & Helander, Phys. Plasmas 21, 042503 (2014)].

Gradient accuracy (measured, documented honestly):
  - Boozer-spectrum -> kinetic <j.B> segment: autodiff vs central FD agree to
    ~3e-6 relative (pure JAX + implicit linear solve).
  - full chain d(objective)/d(boundary dof): the dominant dof agrees with
    central FD to 1.7e-3 at the default resolution (2.5e-3 at CI resolution).
    An FD-step sweep (eps 3e-4 .. 3e-6) shows the FD value converging
    monotonically TOWARD the autodiff value and plateauing at the host
    equilibrium solver's ftol termination-noise floor — the comparison is
    limited by finite differences, not by the autodiff chain.  The vmec_jax
    implicit-adjoint tests document the same FD floor.
  - CPU vs GPU: the objective agrees to ~8e-11 relative at identical inputs.

Expected runtime: the default settings (one warm-start demo, one FD gradient
check, a 25-iteration L-BFGS-B loop) are dominated by the host VMEC
equilibrium solves at ~30-60 s per objective+gradient evaluation, so the loop
runs on the order of half an hour on a laptop CPU (override the count with
SFINCS_JAX_QA_MAXITER).  With SFINCS_JAX_CI=1 everything shrinks to a few
minutes.

Requires the optional companions of this example (not needed by sfincs_jax
itself):  pip install -e /path/to/vmec_jax /path/to/booz_xform_jax

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

from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist
from sfincs_jax.inputs import parse_sfincs_input_text
from sfincs_jax.magnetic_geometry import FluxSurfaceGeometry
from sfincs_jax.phase_space import make_grids
from sfincs_jax.run import profile_moments_from_operator
from sfincs_jax.solve import solve as kinetic_solve

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

try:  # optional companion packages (not needed by sfincs_jax itself)
    import vmec_jax as _vmec_jax_pkg
    from vmec_jax.core import implicit as vmec_implicit
    from vmec_jax.core import optimize as vmec_optimize
    from vmec_jax.core import solver as vmec_solver
    from vmec_jax.core.boozer_tables import boozer_input_tables
    from vmec_jax.core.input import VmecInput
    from vmec_jax.core.wout import wout_from_state, write_wout
    from booz_xform_jax.jax_api import booz_xform_jax as booz_transform
except ImportError as exc:
    raise SystemExit(
        "This example needs vmec_jax (new core API, with core.boozer_tables) and "
        "booz_xform_jax. Install with `pip install -e /path/to/vmec_jax "
        "/path/to/booz_xform_jax`."
    ) from exc

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
CI = os.environ.get("SFINCS_JAX_CI") == "1"  # shrink resolution for CI

# Starting equilibrium: Landreman & Paul (2021) precise QA, low resolution.
# Shipped with vmec_jax (examples/data of an editable checkout); resolved from
# the installed package so no sibling-directory layout is assumed.
VMEC_INPUT = (
    Path(_vmec_jax_pkg.__file__).resolve().parents[1]
    / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"
)
VMEC_INPUT = Path(os.environ.get("SFINCS_JAX_QA_VMEC_INPUT", VMEC_INPUT))

# Equilibrium resolution and convergence (the host VMEC solves dominate cost).
NS = 7 if CI else 13  # radial surfaces
VMEC_FTOL = 1e-11 if CI else 1e-13
VMEC_MAX_ITER = 4000

# Boundary degrees of freedom: RBC/ZBS modes with m,|n| <= MAX_MODE
# (RBC(0,0), the major radius, stays fixed - same convention as simsopt).
MAX_MODE = 1 if CI else 2

# Boozer transform resolution for the kinetic flux surface.
MBOZ, NBOZ = (2, 2) if CI else (3, 3)

# Kinetic solve: two species (ions + electrons), pitch-angle-scattering
# collisions, finite Er with the full-trajectory Er terms (this is what makes
# the system route to the tier-2 GCROT solver, where warm starts/recycling
# apply; it is also a nonsingular, exactly implicit-differentiable system).
KIN_NTHETA, KIN_NZETA, KIN_NXI, KIN_NL, KIN_NX = (9, 7, 8, 4, 4) if CI else (13, 11, 16, 4, 6)
ER = 1.0  # normalized radial electric field (template rHat convention)
NU_N = 0.1  # normalized collisionality
KINETIC_SOLVER_TOL = 1e-9

# Physics targets and penalty weights for the scalar objective.
TARGET_ASPECT = 6.0
TARGET_IOTA = 0.42
QS_SURFACES = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])  # two-term QS residual
W_ASPECT = 1.0
W_IOTA = 100.0
W_QS = 1.0e4
W_KINETIC = 1.0e5

# Kinetic figure of merit entering the objective.  Every entry of the dict is
# CI-tested, so switching the commented line in just works.
KINETIC_OBJECTIVE = "bootstrap_jbs2"  # (<j.B>/sqrt(<B^2>))^2 -> drive to zero
# KINETIC_OBJECTIVE = "particle_flux_l1"  # tested: uncomment to use (L1-style smooth |Gamma_s| sum)
# KINETIC_OBJECTIVE = "heat_flux_l2"      # tested: uncomment to use (sum_s Q_s^2)

# Optimizer (scipy L-BFGS-B on jax.value_and_grad of the objective).  The
# default runs enough iterations to converge the objective; SFINCS_JAX_CI=1
# shrinks it to a couple of iterations for the smoke test.
MAXITER = int(os.environ.get("SFINCS_JAX_QA_MAXITER", "2" if CI else "25"))
BOUND_RADIUS = 0.02  # box bounds |dof - dof0| <= radius keep trial boundaries physical
PENALTY_VALUE = 1.0e6  # returned for trial boundaries where VMEC fails (zero-crash)
RUN_FD_CHECK = os.environ.get("SFINCS_JAX_QA_FD_CHECK", "1") == "1"
# Central-FD step and acceptance gate for the end-to-end gradient check.  An
# eps sweep (3e-4 .. 3e-6) shows the FD value converging monotonically toward
# the autodiff value and plateauing at the host equilibrium solver's ftol
# noise floor (~2e-3 relative at the CI ftol, lower at the default ftol), so
# the comparison below is limited by finite differences, not by autodiff.
FD_EPS = 1e-5
FD_GATE = 5e-2 if CI else 5e-3

OUT_DIR = Path(__file__).parent / "output"
STEM = "optimize_QA_bootstrap"

# ----------------------------------------------------------------------------
# 1) Load the starting boundary; set up the differentiable equilibrium solve
# ----------------------------------------------------------------------------
print("=== examples/optimize_QA_bootstrap.py ===")
print("Step 1: differentiable fixed-boundary equilibrium (vmec_jax.core.implicit)")
if not VMEC_INPUT.exists():
    raise SystemExit(
        f"VMEC input not found: {VMEC_INPUT}\n"
        "Point SFINCS_JAX_QA_VMEC_INPUT at input.LandremanPaul2021_QA_lowres "
        "from the vmec_jax examples/data directory."
    )
inp0 = VmecInput.from_file(str(VMEC_INPUT))
inp0 = dataclasses.replace(
    inp0,
    ns_array=np.asarray([NS]),
    ftol_array=np.asarray([VMEC_FTOL]),
    niter_array=np.asarray([VMEC_MAX_ITER]),
)
cfg = vmec_implicit.make_config(inp0)
params0 = vmec_implicit.params_from_input(inp0)
NFP = int(cfg.resolution.nfp)

dof_modes = vmec_optimize._dof_modes(inp0, MAX_MODE)
NM = len(dof_modes)
NTOR = int(inp0.ntor)
dof_rows = np.asarray([n + NTOR for (_, n) in dof_modes])
dof_cols = np.asarray([m for (m, _) in dof_modes])
dof_names = vmec_optimize.boundary_dof_names(inp0, MAX_MODE)
dofs0 = jnp.asarray(vmec_optimize.pack_boundary(inp0, MAX_MODE))
print(f"  starting point: {VMEC_INPUT.name} (nfp={NFP}, ns={NS}, ftol={VMEC_FTOL:g})")
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

# ----------------------------------------------------------------------------
# 2) Kinetic operator template (built once; geometry replaced per evaluation)
# ----------------------------------------------------------------------------
# The template fixes species, collisionality, Er and grids.  psiAHat/aHat only
# set the rHat/psiHat conversions for the (frozen) input gradients; they are
# taken from the starting equilibrium and are not re-derived per iteration
# (standard practice: plasma profiles stay fixed while the boundary moves).
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
# 3) The differentiable physics chain, written out in this script
# ----------------------------------------------------------------------------
# (a) Traceable single-surface VMEC spectral tables from the solved state:
# the glue between the two codes.  It evaluates the vmec_jax core field chain
# (pure JAX), mirrors the reduced [0, pi] theta grid to the full circle with
# stellarator symmetry, and projects onto the wout cos(m*theta - n*zeta) /
# sin(...) mode tables that booz_xform_jax consumes.
# tests/test_example_qa_bootstrap.py validates every table against the host
# wout engine (bmnc/rmnc/zmns to ~1e-15; bsub*/lmns to ~1e-3, the half-mesh
# finite-difference level) and the resulting Boozer |B| spectrum against the
# classic host booz_xform run (~3e-6).
# The helper is a public vmec_jax core function (vmec_jax.core.boozer_tables,
# added by uwplasma/vmec_jax PR #23; imported at the top of this script).

# (b) Boozer |B| spectrum -> canonical kinetic solve -> per-species moments.
# FluxSurfaceGeometry.from_fourier is the pure-JAX geometry entry point
# (geometryScheme 13); the geometry leaves of the operator pytree are swapped
# with dataclasses.replace, so the whole operator is traced.  BBar = 1 T and
# RBar = 1 m, hence the Boozer amplitudes/G/I feed in unchanged.


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


# (c) Kinetic figures of merit.  All entries are evaluated by the CI test, so
# switching KINETIC_OBJECTIVE at the top of the file just works.
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


# (d) The scalar objective: a plain JAX function of the boundary dofs.
def objective(dofs, warm=None):
    """Total objective and diagnostics dict; differentiable in ``dofs``."""
    rbc = params0.rbc.at[dof_rows, dof_cols].set(dofs[:NM])
    zbs = params0.zbs.at[dof_rows, dof_cols].set(dofs[NM:])
    params = dataclasses.replace(params0, rbc=rbc, zbs=zbs)
    state = vmec_implicit.solve_implicit(params, cfg)  # custom-VJP equilibrium
    rt = vmec_implicit.runtime_from_params(params, cfg)

    aspect = vmec_optimize.aspect_ratio(state, rt)
    iota_mean = vmec_optimize.mean_iota(state, rt)
    qs = qs_metric.total_state(state, rt)

    tabs = boozer_input_tables(state, rt, S_KINETIC_ROW)
    booz = booz_transform(
        rmnc=tabs["rmnc"][None, :], zmns=tabs["zmns"][None, :], lmns=tabs["lmns"][None, :],
        bmnc=tabs["bmnc"][None, :], bsubumnc=tabs["bsubumnc"][None, :],
        bsubvmnc=tabs["bsubvmnc"][None, :], iota=tabs["iota"][None],
        xm=tabs["xm"], xn=tabs["xn"], xm_nyq=tabs["xm"], xn_nyq=tabs["xn"],
        nfp=NFP, mboz=MBOZ, nboz=NBOZ, asym=False,
    )
    mom, result = kinetic_moments(
        booz,
        x0=None if warm is None else warm.get("x0"),
        recycle=None if warm is None else warm.get("recycle"),
    )
    kinetic_term = KINETIC_OBJECTIVES[KINETIC_OBJECTIVE](mom)
    jbs = mom["FSABjHatOverRootFSAB2"]

    total = (W_ASPECT * (aspect - TARGET_ASPECT) ** 2
             + W_IOTA * (iota_mean - TARGET_IOTA) ** 2
             + W_QS * qs
             + W_KINETIC * kinetic_term)
    sg = jax.lax.stop_gradient
    aux = {
        "aspect": aspect, "iota": iota_mean, "qs": qs, "jbs": jbs,
        "kinetic_term": kinetic_term,
        # Boozer-surface snapshot (used by the plot and by the tests'
        # kinetic-segment gradient check):
        "bmnc_b": sg(booz["bmnc_b"][0]),
        "booz_iota": sg(booz["iota_b"][0]),
        "booz_G": sg(booz["bvco_b"][0]),
        "booz_I": sg(booz["buco_b"][0]),
        # per-species kinetic moments feeding the alternative objectives:
        "particle_flux": sg(mom["particleFlux_vm_psiHat"]),
        "heat_flux": sg(mom["heatFlux_vm_psiHat"]),
        # kinetic warm-start state for the next evaluation:
        "x_solution": sg(result.x),
        "recycle": (None if result.recycle is None
                    else tuple(sg(r) for r in result.recycle)),
        "kinetic_iterations": result.iterations,  # None under tracing
    }
    return total, aux


# ----------------------------------------------------------------------------
# 4) Initial conditions + warm-start savings of the kinetic Krylov solve
# ----------------------------------------------------------------------------
print("Step 3: initial evaluation (cold kinetic solve)")
t0 = time.perf_counter()
J0, aux0 = objective(dofs0)
t_first = time.perf_counter() - t0
warm_state = {"x0": aux0["x_solution"], "recycle": aux0["recycle"]}
print(f"  objective J      = {float(J0):.6e}   ({t_first:.1f} s incl. JIT)")
print(f"  aspect ratio     = {float(aux0['aspect']):.4f} (target {TARGET_ASPECT})")
print(f"  mean iota        = {float(aux0['iota']):.4f} (target {TARGET_IOTA})")
print(f"  QS residual      = {float(aux0['qs']):.4e}")
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
# 5) End-to-end gradient, checked against central finite differences
# ----------------------------------------------------------------------------
print("Step 5: jax.value_and_grad through equilibrium + Boozer + kinetic solve")
value_and_grad = jax.value_and_grad(objective, has_aux=True)
t0 = time.perf_counter()
(J_check, _), grad0 = value_and_grad(dofs0, warm_state)
t_grad = time.perf_counter() - t0
grad0_np = np.asarray(grad0)
print(f"  |grad| = {np.linalg.norm(grad0_np):.4e}   ({t_grad:.1f} s)")

fd_check = None
if RUN_FD_CHECK:
    k_fd = int(np.argmax(np.abs(grad0_np)))  # dominant dof
    vp, _ = objective(dofs0.at[k_fd].add(FD_EPS), warm_state)
    vm, _ = objective(dofs0.at[k_fd].add(-FD_EPS), warm_state)
    fd = (float(vp) - float(vm)) / (2.0 * FD_EPS)
    rel = abs(grad0_np[k_fd] - fd) / max(abs(fd), 1e-300)
    fd_check = {"dof": k_fd, "name": dof_names[k_fd],
                "ad": float(grad0_np[k_fd]), "fd": fd, "rel": rel}
    print(f"  FD check on dof {k_fd} ({dof_names[k_fd]}): "
          f"AD={grad0_np[k_fd]:.8e} FD={fd:.8e} rel={rel:.2e}")
    print("  (the Boozer->kinetic segment alone is accurate to ~1e-6; the full")
    print("   chain is limited by the host equilibrium solver's ftol noise in FD)")
    if not (np.isfinite(fd) and rel < FD_GATE):
        raise SystemExit(f"end-to-end gradient check FAILED: rel {rel:.3e}")

# ----------------------------------------------------------------------------
# 6) Optimize: scipy L-BFGS-B on the JAX value-and-gradient
# ----------------------------------------------------------------------------
print(f"Step 6: L-BFGS-B, {MAXITER} iterations, kinetic objective '{KINETIC_OBJECTIVE}'")
history = []
_eval_index = [0]


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
        "jbs": float(aux["jbs"]),
        "wall_s": time.perf_counter() - t_start,
    }
    history.append(record)
    print(f"  eval {record['eval']:3d}: J={record['objective']:.6e} "
          f"qs={record['qs']:.3e} <j.B>={record['jbs']:+.4e} "
          f"aspect={record['aspect']:.3f} iota={record['iota']:.4f} "
          f"({record['wall_s']:.1f} s)")
    return float(value), np.asarray(grad, dtype=float)


bounds = [(float(v) - BOUND_RADIUS, float(v) + BOUND_RADIUS) for v in np.asarray(dofs0)]
opt = scipy.optimize.minimize(
    scipy_fun, np.asarray(dofs0, dtype=float), jac=True, method="L-BFGS-B",
    bounds=bounds, options={"maxiter": MAXITER, "maxcor": 10},
)
dofs_final = jnp.asarray(opt.x)
J_final, aux_final = objective(dofs_final, warm_state)
print(f"  optimizer stop: {opt.message}")

# ----------------------------------------------------------------------------
# 7) Final results, saved outputs (optimized input + wout), read-back, plot
# ----------------------------------------------------------------------------
print("Step 7: final results")
print(f"  objective        {float(J0):.6e} -> {float(J_final):.6e}")
print(f"  <j.B>/sqrt(<B^2>) {float(aux0['jbs']):+.6e} -> {float(aux_final['jbs']):+.6e}")
print(f"  QS residual      {float(aux0['qs']):.4e} -> {float(aux_final['qs']):.4e}")
print(f"  aspect ratio     {float(aux0['aspect']):.4f} -> {float(aux_final['aspect']):.4f}")
print(f"  mean iota        {float(aux0['iota']):.4f} -> {float(aux_final['iota']):.4f}")
objective_decreased = float(J_final) < float(J0)
print(f"  objective decreased: {objective_decreased}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
inp_final = vmec_optimize.unpack_boundary(inp0, np.asarray(dofs_final, dtype=float), MAX_MODE)
input_path = inp_final.to_indata(OUT_DIR / f"input.{STEM}_optimized")

# wout-equivalent of the optimized boundary via the host solver (also the
# "before" file for the plot comparison)
wout_paths = {}
for tag, inp_tag in (("initial", inp0), ("final", inp_final)):
    res = vmec_solver.solve(inp_tag, cfg.resolution, ftol=cfg.ftol,
                            max_iterations=cfg.max_iterations, mode="cli")
    wout = wout_from_state(inp=inp_tag, state=res.state, fsqr=res.fsqr,
                           fsqz=res.fsqz, fsql=res.fsql)
    wout_paths[tag] = write_wout(OUT_DIR / f"wout_{STEM}_{tag}.nc", wout)

history_path = OUT_DIR / f"{STEM}_history.json"
history_path.write_text(json.dumps({
    "history": history,
    "initial": {k: float(aux0[k]) for k in ("aspect", "iota", "qs", "jbs")},
    "final": {k: float(aux_final[k]) for k in ("aspect", "iota", "qs", "jbs")},
    "objective_initial": float(J0), "objective_final": float(J_final),
    "warm_start": {"iterations_cold": it_cold, "iterations_warm": it_warm,
                   "seconds_first": t_first, "seconds_warm": t_warm},
    "kinetic_objective": KINETIC_OBJECTIVE,
    "dof_names": dof_names,
}, indent=2) + "\n")
read_back = json.loads(history_path.read_text())
print(f"  read back from json: objective_final = {read_back['objective_final']:.6e}")

# Before/after |B| on the kinetic Boozer surface + convergence history.
def _bmag_boozer(bmnc_b, ixm, ixn, ntheta=60, nzeta=60):
    th = np.linspace(0, 2 * np.pi, ntheta)
    ze = np.linspace(0, 2 * np.pi / NFP, nzeta)
    ang = th[:, None, None] * ixm[None, None, :] - ze[None, :, None] * ixn[None, None, :]
    return th, ze, np.einsum("m,tzm->tz", np.asarray(bmnc_b), np.cos(ang))


fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
for ax, (tag, aux_tag) in zip(axes[0], (("initial", aux0), ("final", aux_final))):
    th, ze, bmag = _bmag_boozer(np.asarray(aux_tag["bmnc_b"]), BOOZ_XM, BOOZ_XN)
    im = ax.contourf(ze, th, bmag, levels=24, cmap="viridis")
    ax.set_title(f"|B| (T), Boozer angles, s~{S_KINETIC:.2f} ({tag})")
    ax.set_xlabel("zeta_B")
    ax.set_ylabel("theta_B")
    fig.colorbar(im, ax=ax, fraction=0.046)

evals = [h["eval"] for h in history]
axes[1, 0].semilogy(evals, [h["objective"] for h in history], "o-")
axes[1, 0].set_xlabel("objective evaluation")
axes[1, 0].set_ylabel("total objective")
axes[1, 0].set_title("optimization progress")
axes[1, 1].semilogy(evals, [abs(h["jbs"]) for h in history], "o-", label="|<j.B>|/sqrt(<B^2>)")
axes[1, 1].semilogy(evals, [h["qs"] for h in history], "s-", label="QS residual")
axes[1, 1].set_xlabel("objective evaluation")
axes[1, 1].legend()
axes[1, 1].set_title("bootstrap current and quasisymmetry")
fig.tight_layout()
plot_path = OUT_DIR / f"{STEM}.png"
fig.savefig(plot_path, dpi=130)
plt.close(fig)

print(f"  Saved plot: {plot_path}")
print(f"  Wrote output files: {input_path.name}, "
      f"{wout_paths['initial'].name}, {wout_paths['final'].name}, {history_path.name}")
print("Done: examples/optimize_QA_bootstrap.py")
