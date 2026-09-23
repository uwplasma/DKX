"""A stellarator boundary optimized with a kinetic bootstrap-current row.

VMEX's examples/optimization/QA_optimization.py -- same seed deck, same mode
ladder, same SciPy least_squares driver -- with one residual row added: the
bootstrap current <j.B> that DKX computes by solving the drift-kinetic
equation on one flux surface of every trial equilibrium.  The chain boundary
coefficients -> VMEX -> booz_xform_jax -> DKX is traced end to end, so VMEX's
exact implicit Jacobian carries the DKX row like any other; the analytic Redl
<j.B> for the same surface and profiles is a zero-weight row beside it.

Physics: vacuum nfp = 2 quasi-axisymmetric equilibrium; ions and electrons
with n = n0 (1 - s^5), Te = Ti = T0 (1 - s), whose pressure is not fed back
to the field; E_r = 0; banana regime (nu* ~ 0.07).  Pitch-angle scattering
on the exact structured solve: cheap and exactly differentiable, but it does
not conserve momentum, so <j.B> is the PAS estimate, not Fokker-Planck.

Needs the optional packages vmex and booz_xform_jax.  No case.toml: an
optimization is not a case.  The kinetic grid is a teaching grid; see
docs/experiments/2026-09-23-vmex-kinetic-optimization.md for refinement.
Expected runtime: about ten minutes on a laptop CPU, most of it compilation.
"""

# 1. Imports
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from booz_xform_jax.jax_api import (  # noqa: E402
    BoozerConfig,
    booz_xform_jax_impl,
    prepare_booz_xform_plan,
)
from netCDF4 import Dataset  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import dkx  # noqa: E402
import vmex as vj  # noqa: E402
from dkx.run import profile_moments_from_operator  # noqa: E402
from dkx.solve import solve  # noqa: E402
from dkx.units import PARALLEL_CURRENT  # noqa: E402
from dkx.workflows.geometry_adapters import (  # noqa: E402
    boozer_route_psi_a_hat,
    kinetic_operator_on_boozer_surface,
)
from vmex import optimize as opt  # noqa: E402
from vmex.core.boozer_tables import boozer_input_tables  # noqa: E402
from vmex.core.bootstrap import (  # noqa: E402
    KineticProfiles,
    j_dot_B_redl,
    redl_geometry_from_state,
)

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "08_vmex_optimization"
RESULT_FILE = OUT_DIR / "optimization.nc"
PLOT_FILE = OUT_DIR / "optimization.png"
INPUT_FILE = HERE / "input.minimal_seed_nfp2"  # vmex's examples/data seed deck

# QA_optimization.py, verbatim except the ladder, which is one stage here so
# the laptop run stays in minutes.  Its full ladder is MAX_MODES = [1, 2, 3],
# MAX_NFEV = [10, 10, 15].
SEED_PERTURBATION = 0.05
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES = [1]
MAX_NFEV = [8]
ASPECT_TARGET = 5.0
MAGNETIC_WELL_TARGET = 0.01
IOTA_FLOOR = 0.42
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2
VARY_MAJOR_RADIUS = False
MINIMUM_MPOL = 5
FINAL_NS = 101
FINAL_FTOL = 1.0e-14
FINAL_NITER = 8000

# The kinetic row.  <j.B> is in MA T/m^2, the unit VMEX plots jdotb in.
# Target 0 asks for a lower bootstrap current at the kinetic surface.
BOOTSTRAP_TARGET = 0.0
BOOTSTRAP_WEIGHT = 1.0
# Zero weight: the Redl row is reported, not optimized.  A positive weight
# would optimize the analytic estimate alongside the kinetic one.
REDL_WEIGHT = 0.0
N0, T0 = 1.0e19, 2.0e3  # m^-3 and eV on axis
KINETIC_GRID = dict(Ntheta=11, Nzeta=11, Nxi=16, NL=4, Nx=4)
COLLISION_OPERATOR = 1  # 0 (Fokker-Planck) needs the Krylov route; see the record
MBOZ = NBOZ = 6
KINETIC_TOLERANCE = 1.0e-10
# Central differences for the derivative check, one step per coefficient.
FD_COEFFICIENTS = ("RBC(1,0)", "RBC(0,1)", "ZBS(-1,1)")
FD_STEP = 1.0e-4
# Extension point, not implemented: the thermal transport coefficients
# Le1/Li1 (L11 of Lascas Neto, Jorge, Beidler & Lion, JPP 91 E24, 2025) from
# RHSMode = 2.  They join as further rows beside the bootstrap row once a
# test pins the paper's mass factor (plan.md section 13, step 9).
TRANSPORT_COEFFICIENT_ROWS = ()
# end of parameters

# 3. Geometry and species construction
OUT_DIR.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_enable_x64", True)
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
mpol = max(max(MAX_MODES) + 2, MINIMUM_MPOL)
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
NFP = int(inp.nfp)
NS = int(inp.ns_array[-1])
KINETIC_ROW = NS // 2  # half-mesh row of the kinetic surface
S_KINETIC = (KINETIC_ROW - 0.5) / (NS - 1)
equilibrium = opt.solve_equilibrium(inp)
seed_equilibrium = equilibrium

# The same profiles feed Redl (in SI) and DKX (in its nBar = 1e20 m^-3,
# TBar = 1 keV units, with gradients in d/dpsiN so the minor radius, which the
# optimizer changes, never enters).
PROFILES = KineticProfiles(N0 * np.array([1, 0, 0, 0, 0, -1.0]),
                           T0 * np.array([1, -1.0]), T0 * np.array([1, -1.0]))
n_hat, t_hat = N0 / 1e20 * (1 - S_KINETIC**5), T0 / 1e3 * (1 - S_KINETIC)
dn_hat, dt_hat = -5 * N0 / 1e20 * S_KINETIC**4, -T0 / 1e3

# The Boozer transform's mode tables are built once on the host, so the traced
# transform inside the Jacobian is pure arithmetic; its first call fixes the
# output mode list and the handedness.
tables = boozer_input_tables(equilibrium.state, equilibrium.runtime, KINETIC_ROW)
xm, xn = np.asarray(tables["xm"]), np.asarray(tables["xn"])
plan = prepare_booz_xform_plan(nfp=NFP, asym=False, xm=xm, xn=xn, xm_nyq=xm, xn_nyq=xn,
                               config=BoozerConfig.from_env(mboz=MBOZ, nboz=NBOZ))
booz_modes = dict(xm=jnp.asarray(xm), xn=jnp.asarray(xn), xm_nyq=jnp.asarray(xm),
                  xn_nyq=jnp.asarray(xn), constants=plan.constants, grids=plan.grids, plan=plan)
booz0 = booz_xform_jax_impl(**{k: jnp.asarray(tables[k])[None] for k in (
    "rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota")}, **booz_modes)
IXM_B, IXN_B = np.asarray(booz0["ixm_b"]), np.asarray(booz0["ixn_b"])
psi_a_hat = boozer_route_psi_a_hat(
    float(np.asarray(equilibrium.wout.phi)[-1]), int(equilibrium.wout.signgs),
    float(booz0["bvco_b"][0] + booz0["iota_b"][0] * booz0["buco_b"][0]))

# 4. Physics and numerical configuration
template = dkx.run(
    geometryScheme=1, psiAHat=psi_a_hat, aHat=0.1,  # geometry replaced below
    inputRadialCoordinate=1, inputRadialCoordinateForGradients=1, psiN_wish=S_KINETIC,
    Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4], nHats=[n_hat, n_hat], THats=[t_hat, t_hat],
    dNHatdpsiNs=[dn_hat, dn_hat], dTHatdpsiNs=[dt_hat, dt_hat],
    collisionOperator=COLLISION_OPERATOR, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3, Er=0.0,
    **KINETIC_GRID, emit=None,
).operator  # fmt: skip
print(f"kinetic surface: half-mesh row {KINETIC_ROW} of {NS}, s = {S_KINETIC:.4f}; "
      f"operator size {template.total_size}; psiAHat = {psi_a_hat:+.5e} (signgs "
      f"{int(equilibrium.wout.signgs):+d})")


def kinetic_bootstrap(equilibrium_state, solver_context, certificate=False):
    """DKX <j.B> [MA T/m^2] on the kinetic surface.  Traced, so VMEX differentiates it."""
    surface = boozer_input_tables(equilibrium_state, solver_context, KINETIC_ROW)
    booz = booz_xform_jax_impl(**{k: jnp.asarray(surface[k])[None] for k in (
        "rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota")}, **booz_modes)
    operator = kinetic_operator_on_boozer_surface(
        template, bmnc_b=booz["bmnc_b"][0], ixm_b=IXM_B, ixn_b=IXN_B, nfp=NFP,
        iota=booz["iota_b"][0], g_hat=booz["bvco_b"][0], i_hat=booz["buco_b"][0])
    rhs = operator.rhs()
    # tier1_keep_lowest = Nxi solves every Legendre block, so the solution
    # satisfies the original equation, not only its lowest three blocks.
    solved = solve(operator, rhs, method="auto", tol=KINETIC_TOLERANCE, differentiable=True,
                   tier1_keep_lowest=operator.n_xi, emit=None)  # fmt: skip
    x = solved.x.reshape(-1)
    current = profile_moments_from_operator(operator, x)["FSABjHat"].reshape(())
    current = current * PARALLEL_CURRENT / 1e6
    if not certificate:
        return current
    residual = jnp.linalg.norm(operator.apply(x) - rhs.reshape(-1)) / jnp.linalg.norm(rhs)
    return current, residual, booz["bmnc_b"][0]


def redl_bootstrap(equilibrium_state, solver_context):
    """Redl's analytic <j.B> [MA T/m^2] for the same surface and profiles."""
    geometry = redl_geometry_from_state(
        equilibrium_state, solver_context, surfaces=np.array([S_KINETIC]))
    return j_dot_B_redl(PROFILES, geometry, 0)[0][0] / 1e6


def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|, as in QA_optimization.py."""
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
    (kinetic_bootstrap, BOOTSTRAP_TARGET, BOOTSTRAP_WEIGHT),
    (redl_bootstrap, 0.0, REDL_WEIGHT),
    *TRANSPORT_COEFFICIENT_ROWS,
]
report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".4e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("well", opt.magnetic_well, ".4f"),
    ("<j.B> DKX", kinetic_bootstrap, "+.5f"), ("<j.B> Redl", redl_bootstrap, "+.5f"))
# The finer re-solve has a different radial grid, so the kinetic row (tied to
# a half-mesh row of the optimizer's grid) is quoted on the optimizer's grid.
final_report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".4e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("well", opt.magnetic_well, ".4f"),
    ("<j.B> Redl", redl_bootstrap, "+.5f"))
monitor = opt.OptimizationMonitor(stream=None)

# 5. Run
clock = time.perf_counter()
problem = opt.VmecProblem.from_tuples(
    inp, objective_function_terms, max_mode=max(MAX_MODES),
    vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ESS_ALPHA,
    restart_from=equilibrium)
monitor.problem = problem
dof_names = list(problem.dof_names)
row = {name: start for name, start, _ in problem.metadata["term_slices"]}["kinetic_bootstrap"]
jacobian0 = problem.residual_jac(problem.x0)
setup_seconds = time.perf_counter() - clock
print(f"first residual and Jacobian (compilation included): {setup_seconds:.0f} s")

# The admission check: the DKX row of VMEX's Jacobian against central
# differences of the same residual, one full equilibrium re-solve per point.
scale = np.sqrt(BOOTSTRAP_WEIGHT)
fd_rows = []
for name in FD_COEFFICIENTS:
    k = dof_names.index(name)
    shift = FD_STEP * np.eye(len(dof_names))[k]  # x holds the coefficients themselves [m]
    clock = time.perf_counter()
    plus = problem.residual(problem.x0 + shift)[row]
    residual_seconds = time.perf_counter() - clock  # one warm residual, equilibrium included
    minus = problem.residual(problem.x0 - shift)[row]
    fd_rows.append((name, float(jacobian0[row, k]) / scale, (plus - minus) / (2 * FD_STEP) / scale))

clock = time.perf_counter()
problem.residual_jac(problem.x0 + 0.5 * FD_STEP)  # a new point, so nothing is cached
jacobian_seconds = time.perf_counter() - clock
cost0 = 0.5 * float(np.sum(problem.residual(problem.x0) ** 2))
clock = time.perf_counter()
x = problem.x0
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QA stage with the kinetic row, max_mode = {max_mode} =====")
    stage = problem.subproblem(max_mode=max_mode, x=x)
    step = PARAMETER_STEP * stage.scales
    result = least_squares(
        stage.residual, stage.x0, jac=stage.residual_jac, x_scale=step, max_nfev=max_nfev,
        bounds=(stage.x0 - MAX_PARAMETER_CHANGE * step, stage.x0 + MAX_PARAMETER_CHANGE * step),
        ftol=1e-6, xtol=1e-10, verbose=1, callback=monitor)
    x = stage.embed(result.x)
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
optimize_seconds = time.perf_counter() - clock

# The optimizer's grid is not the certificate: re-solve the optimized
# boundary on a finer radial grid to a tighter tolerance, as VMEX does.
final_input = replace(inp, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
                      niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
                                          verbose=False, raise_on_max_iterations=True)
before = kinetic_bootstrap(seed_equilibrium.state, seed_equilibrium.runtime, certificate=True)
after = kinetic_bootstrap(equilibrium.state, equilibrium.runtime, certificate=True)
redl_before = float(redl_bootstrap(seed_equilibrium.state, seed_equilibrium.runtime))
redl_after = float(redl_bootstrap(equilibrium.state, equilibrium.runtime))

# 6. Print a scientific summary and certificate
costs = np.array([cost0] + [record.cost for record in monitor.records])
print("\n=== Final results ===")
print("d<j.B>_DKX / d(coefficient) [MA T/m^2 per m]: VMEX implicit Jacobian vs central FD")
worst = 0.0
for name, exact, central in fd_rows:
    relative = abs(exact - central) / max(abs(exact), abs(central), 1e-300)
    worst = max(worst, relative)
    print(f"  {name:>10s}: Jacobian {exact:+.6e}   FD {central:+.6e}   rel. diff {relative:.1e}")
assert worst <= 1e-2, "the DKX row of the Jacobian disagrees with finite differences"
print(f"  Jacobian row verified against central differences (worst {worst:.1e} <= 1e-2)")
print(f"objective 0.5*|r|^2: {costs[0]:.6e} -> {costs[-1]:.6e} over {len(costs) - 1} iterations")
assert costs[-1] < costs[0], "the optimization did not lower its objective"
report("seed", seed_equilibrium)
report("final (optimizer grid)", equilibrium)
final_report(f"final (ns = {FINAL_NS}, ftol = {FINAL_FTOL:g})", final_equilibrium)
print(f"VMEX re-solve converged at ns = {FINAL_NS}: fsq within ftol = {FINAL_FTOL:g}")
print(f"<j.B> DKX  seed {float(before[0]):+.5f}  final {float(after[0]):+.5f} MA T/m^2 "
      f"(original-equation residual {float(before[1]):.1e}, {float(after[1]):.1e})")
print(f"<j.B> Redl seed {redl_before:+.5f}  final {redl_after:+.5f} MA T/m^2 "
      f"(DKX/Redl {float(before[0]) / redl_before:.3f} -> {float(after[0]) / redl_after:.3f})")
assert max(float(before[1]), float(after[1])) < 1e-8, "a DKX solve missed its original equation"
print(f"wall time: {setup_seconds:.0f} s first residual and Jacobian (compilation), then "
      f"{residual_seconds:.1f} s per residual, {jacobian_seconds:.1f} s per Jacobian, "
      f"{optimize_seconds:.0f} s for the optimization")

# 7. Save native result
with Dataset(RESULT_FILE, "w", format="NETCDF4") as dataset:
    dataset.createDimension("iteration", len(costs))
    dataset.createDimension("coefficient", len(fd_rows))
    dataset.createDimension("mode", len(IXM_B))
    dataset.createVariable("cost", "f8", ("iteration",))[:] = costs
    dataset.createVariable("jacobian_dkx_row", "f8", ("coefficient",))[:] = [r[1] for r in fd_rows]
    dataset.createVariable("central_difference_dkx_row", "f8", ("coefficient",))[:] = [
        r[2] for r in fd_rows]
    for label, values in (("bmnc_b_seed", before[2]), ("bmnc_b_final", after[2])):
        dataset.createVariable(label, "f8", ("mode",))[:] = np.asarray(values)
    for label, value in (("jdotb_dkx_seed", before[0]), ("jdotb_dkx_final", after[0]),
                         ("jdotb_redl_seed", redl_before), ("jdotb_redl_final", redl_after),
                         ("residual_dkx_seed", before[1]), ("residual_dkx_final", after[1])):
        dataset.createVariable(label, "f8")[...] = float(value)
    dataset.coefficients = ", ".join(r[0] for r in fd_rows)
    dataset.s_kinetic = S_KINETIC
    dataset.dkx_version = dkx.__version__
    dataset.vmex_version = vj.__version__
print(f"  Wrote result: {RESULT_FILE}")
print(f"  Wrote {final_input.to_indata(str(OUT_DIR / 'input.QA_dkx_bootstrap'))}")
print(f"  Wrote {vj.write_wout(str(OUT_DIR / 'wout_QA_dkx_bootstrap.nc'), final_equilibrium.wout)}")

# 8. Plot publication-ready outputs
theta = np.linspace(0, 2 * np.pi, 64)
zeta = np.linspace(0, 2 * np.pi / NFP, 64)
angle = IXM_B[:, None, None] * theta[None, :, None] - IXN_B[:, None, None] * zeta[None, None, :]
figure, axes = plt.subplots(1, 4, figsize=(15.0, 3.4), constrained_layout=True)
axes[0].semilogy(costs, "o-", color="tab:blue")
axes[0].set(xlabel="least-squares iteration", ylabel=r"$\frac{1}{2}|r|^2$",
            title="objective, kinetic row included")
labels = ("seed", "final")
positions = np.arange(2)
axes[1].bar(positions - 0.18, [float(before[0]), float(after[0])], 0.36, label="DKX (PAS)")
axes[1].bar(positions + 0.18, [redl_before, redl_after], 0.36, label="Redl")
axes[1].set(xticks=positions, xticklabels=labels, ylabel=r"$\langle j\cdot B\rangle$ [MA T/m$^2$]",
            title=f"bootstrap current at s = {S_KINETIC:.2f}")
axes[1].legend(frameon=False)
for axis, spectrum, label in ((axes[2], before[2], "seed"), (axes[3], after[2], "final")):
    field = np.sum(np.asarray(spectrum)[:, None, None] * np.cos(angle), axis=0)
    image = axis.contourf(zeta, theta, field, 24, cmap="viridis")
    axis.set(xlabel=r"$\zeta_B$", ylabel=r"$\theta_B$", title=f"|B| [T] on s = {S_KINETIC:.2f}, {label}")
    figure.colorbar(image, ax=axis)
figure.savefig(PLOT_FILE, dpi=90)
plt.close(figure)
print(f"  Saved plot: {PLOT_FILE}")
print("Done: examples/08_vmex_optimization/run.py")
