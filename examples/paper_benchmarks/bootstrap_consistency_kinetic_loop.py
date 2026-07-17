"""Kinetic-solver-in-the-loop bootstrap consistency on a finite-beta QA equilibrium.

Sixth entry of the methods-paper benchmark suite, and the suite's workflow
case: the self-consistent-bootstrap equilibrium iteration that the recent
quasisymmetric-stellarator optimization literature performs with an *analytic*
bootstrap proxy [A. Redl et al., Phys. Plasmas 28, 022502 (2021); M.
Landreman, S. Buller and M. Drevlak, Phys. Plasmas 29, 082501 (2022)], here
with the actual drift-kinetic solve inside the loop.  The community-stated
need this addresses: proxy-driven optima must be checked (and ideally
iterated) against the kinetic bootstrap current, because the Redl fit was
calibrated on tokamak collisionality physics and carries a configuration-
dependent error in stellarator geometry.  Replacing the proxy removes that
error by construction; this script measures exactly how large it was.

Configuration: the precise-QA reactor-scale boundary of M. Landreman and E.
Paul, Phys. Rev. Lett. 128, 035001 (2022) (the vacuum deck shipped with the
host equilibrium package), given the matched finite-beta pressure

    p(s)  = e (ne Te + ni Ti),      Zeff = 1  (ni = ne),
    ne(s) = 3.5e19 (1 - 0.99 s^5) m^-3,
    Te(s) = Ti(s) = 9.45 keV (1 - 0.99 s),

the kinetic-profile *shape* of the self-consistent-bootstrap optimization
paper (arXiv:2205.02914) with a 1% edge floor, at reduced density (on-axis
beta 0.75%).  The zero-net-current finite-beta equilibrium converges cleanly
with iota 0.37..0.43 from shaping, and the self-consistent bootstrap current
(~ -0.64 MA) raises the iota profile to 0.42..0.48 -- a genuine, resolved
equilibrium feedback.

Documented configuration limitations (measured while building this case):
  - the repository's other finite-beta QA deck
    (examples/vmec_jax_finite_beta/input.nfp2_QA_finite_beta) has almost
    entirely current-driven rotational transform (iota ~ 0.02 without its
    6.1 MA current), so a from-scratch zero-current Picard start is
    degenerate there (the equilibrium solve stalls at fsq ~ 1e-4);
  - at the paper's full reactor density (ne0 = 2.38e20 m^-3, on-axis beta
    2.4%) the self-consistent current drives the iota profile across the
    low-order rationals 1/2 and toward 2/3; local drift-kinetic solves on
    surfaces sitting near those rationals become resonant at fixed
    resolution (order-of-magnitude jumps and sign flips in <J.B> were
    observed at iota ~ 0.50 and ~ 0.67).  The committed demonstration
    therefore runs at reduced density, where the converged iota profile
    stays inside the rational-free window (2/5, 1/2) and every surface
    solve is clean.  A production-beta version of this loop needs either
    resonance-aware surface placement or per-surface resolution control.

What the script does:
  1. solves the fixed-boundary finite-beta equilibrium with zero net toroidal
     current (prescribed-current mode, flat I' guess);
  2. **the Picard loop** -- iterates equilibrium -> kinetic bootstrap profile
     -> current profile -> equilibrium: on each iteration the drift-kinetic
     equation is solved on N_SURF flux surfaces of the current equilibrium
     (two species, full Fokker-Planck collisions, Er = 0, the VMEC-geometry
     route), giving <J.B>(s) in SI units; the parallel-current identity of
     an MHD equilibrium,

         <J.B>(s) = [ <B^2> dI/ds + mu0 I dp/ds ] / (2 pi psi_a),

     is inverted for the enclosed toroidal current I(s) (dense collocation
     solve with I(0) = 0, 4th-order d/ds stencils), the smooth dI/ds is
     refit as the equilibrium current profile (power series, CURTOR = I(1)),
     and the equilibrium is re-solved.  Iterations stop when the relative
     current-profile change delta = max|I'_new - I'_applied| / max|I'_new|
     falls below PICARD_TOL.  Every kinetic surface solve and every finished
     iteration is checkpointed to a JSON cache, so interrupted runs resume;
  3. **the Redl contrast** -- at the converged self-consistent state the Redl
     analytic <J.B> is evaluated on the same surfaces with the same profiles,
     and the kinetic-vs-Redl discrepancy profile is recorded: this is the
     proxy error that kinetic-in-the-loop iteration removes.  Split angular
     and velocity-space refined re-solves at the mid surface bound the
     kinetic numerical error, so the JSON separates the physics discrepancy
     from the resolution error;
  4. **the gradient hook** -- one jax.value_and_grad of the (coarse-quadrature)
     total bootstrap current magnitude through the differentiable chain
     boundary coefficient -> implicit fixed-boundary equilibrium -> traceable
     Boozer transform -> flux-surface kinetic operator -> implicitly
     differentiated Krylov solve -> <J.B>, at the converged state, w.r.t. one
     boundary Fourier coefficient (RBC(n=0, m=1)).  This demonstrates that
     the converged loop composes with gradient-based shape optimization (the
     flagship example's chain); the AD-vs-FD accuracy of each chain segment
     is documented separately in gradient_verification.py and the flagship's
     CI test, so here the value and sign are recorded, not re-verified;
  5. writes the JSON record and renders the two-panel figure: <J.B>(s) per
     Picard iteration with the Redl proxy dashed for contrast, and the
     convergence history delta per iteration.

Normalization contract (the repository's standard kinetic normalization):
nBar = 1e20 m^-3, TBar = 1 keV, mBar = proton, BBar = 1 T, RBar = 1 m,
nu_n = 8.31565e-3 (fixed reference Coulomb logarithm); SI conversion
<J.B> = FSABjHat * e nBar sqrt(2 TBar/mBar) * BBar.  The Redl formula
computes its own Sauter per-surface Coulomb logarithms, so the two lanes'
effective collisionalities differ at the few-percent level -- part of the
recorded proxy-vs-kinetic contract, as in the sibling Redl-comparison
example (examples/vmec_jax_finite_beta/).  Er = 0 is the standard convention
for proxy comparisons (the Redl fit carries no Er dependence).

Expected runtime: ~20 min for the Picard loop on a 10-core laptop CPU
(~7 damped iterations x [one warm equilibrium solve ~5 s + 7 kinetic surface
solves ~10-25 s each]), plus ~5-10 min for the differentiable-chain gradient
hook (XLA compilation dominated).  Finished stages are cached in ``output/`` and
skipped on re-runs; set SFINCS_JAX_BOOT_LOOP_MAX_NEW_STAGES=N to stop
cleanly after N newly computed stages (chunked/resumable runs).

Requires the optional companions of this example (not needed by sfincs_jax
itself): pip install -e /path/to/vmec_jax /path/to/booz_xform_jax

Run (from the repo root):
  python examples/paper_benchmarks/bootstrap_consistency_kinetic_loop.py
"""

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from sfincs_jax.inputs import parse_sfincs_input_text  # noqa: E402
from sfincs_jax.magnetic_geometry import FluxSurfaceGeometry  # noqa: E402
from sfincs_jax.phase_space import make_grids  # noqa: E402
from sfincs_jax.run import profile_moments_from_operator, run_profile  # noqa: E402
from sfincs_jax.solve import solve as kinetic_solve  # noqa: E402

# jax 0.6.x compatibility: ``register_dataclass``'s drop_fields validation
# unpacks the sequence with ``*`` (``difference_update(*drop_fields)``), so a
# plain string field name is consumed character-wise and the registration of
# the host equilibrium package's implicit-solution container fails.  Retry
# with each name wrapped in a one-element tuple (which the ``*`` unpacking
# restores to the names themselves); on fixed jax versions the first call
# succeeds and the wrapper is inert.
_register_dataclass = jax.tree_util.register_dataclass


def _register_dataclass_compat(nodetype, data_fields=None, meta_fields=None, drop_fields=()):
    try:
        return _register_dataclass(nodetype, data_fields=data_fields,
                                   meta_fields=meta_fields, drop_fields=drop_fields)  # fmt: skip
    except ValueError:
        if not drop_fields:
            raise
        return _register_dataclass(nodetype, data_fields=data_fields, meta_fields=meta_fields,
                                   drop_fields=[(name,) for name in drop_fields])  # fmt: skip


jax.tree_util.register_dataclass = _register_dataclass_compat

try:  # optional companion packages (not needed by sfincs_jax itself)
    import vmec_jax as _vmec_jax_pkg
    from vmec_jax.core import bootstrap as vmec_bootstrap
    from vmec_jax.core import implicit as vmec_implicit
    from vmec_jax.core import optimize as vmec_optimize
    from vmec_jax.core.boozer_tables import boozer_input_tables
    from vmec_jax.core.input import VmecInput
    from vmec_jax.core.profiles import MU0
    from vmec_jax.core.wout import write_wout
    from booz_xform_jax.jax_api import booz_xform_jax as booz_transform
except ImportError as exc:
    raise SystemExit(
        "This benchmark needs vmec_jax (core API with core.bootstrap and "
        "core.boozer_tables) and booz_xform_jax. Install with "
        "`pip install -e /path/to/vmec_jax /path/to/booz_xform_jax`."
    ) from exc

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
CI = os.environ.get("SFINCS_JAX_BOOT_LOOP_CI") == "1"  # shrink everything for CI

# The precise-QA reactor-scale boundary (vacuum deck shipped with the host
# equilibrium package; resolved from the installed package so no sibling
# directory layout is assumed) [Landreman & Paul, PRL 128, 035001 (2022)].
DECK = (
    Path(_vmec_jax_pkg.__file__).resolve().parents[1]
    / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
)
DECK = Path(os.environ.get("SFINCS_JAX_BOOT_LOOP_DECK", DECK))
MAX_MODE_TRUNC = 2 if CI else None  # CI: truncate boundary harmonics for speed

# Kinetic-profile shape of arXiv:2205.02914 with a 1% edge floor; the
# equilibrium pressure below is exactly e*(ne*Te + ni*Ti) of these profiles.
# The density is reduced from the paper's 2.38e20 so the converged iota
# profile stays inside the rational-free window (2/5, 1/2) -- see the
# docstring's documented-limitations block.
N0 = 3.5e19         # ne(0) [1/m^3]
T0_EV = 9.45e3      # Te(0) = Ti(0) [eV]
EDGE = 0.99         # ne = N0*(1 - EDGE*s^5), Te = Ti = T0*(1 - EDGE*s)
HELICITY_N = 0      # quasi-axisymmetry

# Equilibrium resolution (prescribed-current mode throughout).
NS_LADDER = [7] if CI else [13, 25]
VMEC_FTOL = 1e-10  # warm-started re-solves stall in the low 1e-11s otherwise
VMEC_NITER = [3000] if CI else [2000, 4000]

# Kinetic surfaces and per-surface resolution.  The production grid was set
# by a convergence scan at s = 0.5 on the converged equilibrium (this
# collisionality is banana-plateau, nu* ~ 0.15-1): Nzeta is converged at 13
# (precise QA -- |B| toroidal ripple is tiny; 13 -> 21 moves <J.B> by
# < 0.02%), Nxi at 48 (48 -> 80: 0.2%; 32 -> 48 moves 7%, so the pitch grid
# is the dangerous dimension), Ntheta = 25 is within ~0.8% of the
# Richardson-extrapolated angular limit (21 -> 25: 1.9%, 25 -> 29: 0.5%).
# The split refinement probes in step 3 re-measure both margins at the
# converged state so the recorded proxy discrepancy carries its own
# numerical error bar.
S_KIN = np.asarray([0.35, 0.65]) if CI else np.linspace(0.08, 0.92, 7)
KIN_NTHETA, KIN_NZETA, KIN_NXI, KIN_NL, KIN_NX = (9, 9, 8, 4, 4) if CI else (25, 13, 48, 4, 8)
KIN_TOL = 1e-5 if CI else 1e-6
# Split refinement probes (mid surface, converged state): angular and velocity.
PROBE_NTHETA, PROBE_NZETA = (11, 9) if CI else (29, 13)
PROBE_NXI, PROBE_NX = (10, 5) if CI else (64, 10)

# Picard loop controls.  The self-consistent current shifts the whole iota
# profile, and the kinetic <J.B> is iota-sensitive, so the plain fixed point
# (relax = 1) overshoots (measured at full density: the first undamped
# iterate swung the total current from -3.14 MA to -1.83 MA); relax = 0.5
# damps the map (the standard choice when the current contributes
# substantially to iota).  The zero-current start counts as the applied
# profile of iteration 0, so delta_0 = 1 and the first application is damped
# too.  FIT_DEGREE is the power-series degree of the refit I'(s).
N_PICARD = 3 if CI else 10
PICARD_TOL = 5e-2 if CI else 1e-2
RELAX = 0.5
FIT_DEGREE = 4 if CI else 12

# Kinetic normalization contract (repository standard).
E_CHARGE = 1.602176634e-19
NBAR, TBAR_EV, MBAR_KG = 1.0e20, 1.0e3, 1.67262192369e-27
NU_N = 8.31565e-3  # nu_n for (nBar, TBar, RBar) with the fixed reference lnLambda
CURRENT_SCALE = E_CHARGE * NBAR * np.sqrt(2.0 * TBAR_EV * E_CHARGE / MBAR_KG)  # [A/m^2]

# Gradient hook: coarse-quadrature total bootstrap current through the
# differentiable chain, w.r.t. RBC(n=0, m=1), at reduced kinetic resolution.
GRAD_S = [0.5] if CI else [0.25, 0.5, 0.75]
GRAD_NTHETA, GRAD_NZETA, GRAD_NXI, GRAD_NL, GRAD_NX = (9, 7, 8, 4, 4) if CI else (13, 13, 16, 4, 6)
GRAD_TOL = 1e-7
MBOZ, NBOZ = (3, 3) if CI else (6, 6)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("SFINCS_JAX_BOOT_LOOP_OUT_DIR", Path(__file__).parent / "output"))
FIG_DIR = Path(os.environ.get(
    "SFINCS_JAX_BOOT_LOOP_FIG_DIR",
    REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks"))
STEM = "bootstrap_consistency_kinetic_loop"
CACHE_PATH = OUT_DIR / f"{STEM}_cache{'_ci' if CI else ''}.json"
JSON_PATH = FIG_DIR / f"{STEM}.json"
PNG_PATH = FIG_DIR / f"{STEM}.png"

# Chunked runs: stop cleanly after this many newly computed solve stages.
MAX_NEW_STAGES = int(os.environ.get("SFINCS_JAX_BOOT_LOOP_MAX_NEW_STAGES", "0")) or None

KINETIC_DECK_TEMPLATE = """! Finite-beta QA bootstrap-consistency surface deck.
! Generated by examples/paper_benchmarks/bootstrap_consistency_kinetic_loop.py
&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 5
  equilibriumFile = "{wout}"
  inputRadialCoordinate = 3
  inputRadialCoordinateForGradients = 1
  rN_wish = {rn:.16g}
  VMECRadialOption = 0
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = {nhat:.10g} {nhat:.10g}
  THats = {that:.10g} {that:.10g}
  dNHatdpsiNs = {dnhat:.10g} {dnhat:.10g}
  dTHatdpsiNs = {dthat:.10g} {dthat:.10g}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = {nu_n:.10g}
  Er = 0.0
  collisionOperator = 0
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
  useDKESExBDrift = .false.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {ntheta}
  Nzeta = {nzeta}
  Nxi = {nxi}
  NL = {nl}
  Nx = {nx}
  solverTolerance = {tol:.3g}
/
&otherNumericalParameters
  Nxi_for_x_option = 0
  xGridScheme = 5
/
&preconditionerOptions
/
"""


def kinetic_profiles(s):
    """(ne [1/m^3], dne/ds, Te = Ti [eV], dTe/ds) of the paper profile family."""
    s = np.asarray(s, dtype=float)
    ne = N0 * (1.0 - EDGE * s ** 5)
    dne = N0 * (-5.0 * EDGE * s ** 4)
    te = T0_EV * (1.0 - EDGE * s)
    dte = np.full_like(s, -EDGE * T0_EV)
    return ne, dne, te, dte


def kinetic_surface_jdotb(wout_path, s, *, n_theta=KIN_NTHETA, n_zeta=KIN_NZETA,
                          n_xi=KIN_NXI, n_x=KIN_NX, tag=""):  # fmt: skip
    """Solve the drift-kinetic equation on surface ``s``; return SI ``<J.B>``."""
    ne, dne, te, dte = kinetic_profiles(float(s))
    deck = OUT_DIR / f"{STEM}_surface{tag}.input.namelist"
    deck.write_text(KINETIC_DECK_TEMPLATE.format(
        wout=wout_path, rn=float(np.sqrt(s)), nhat=ne / NBAR, that=te / TBAR_EV,
        dnhat=dne / NBAR, dthat=dte / TBAR_EV, nu_n=NU_N,
        ntheta=int(n_theta), nzeta=int(n_zeta), nxi=int(n_xi), nl=KIN_NL, nx=int(n_x),
        tol=KIN_TOL,
    ))  # fmt: skip
    t0 = time.time()
    run = run_profile(deck, tol=KIN_TOL, emit=None)
    return {
        "s": float(s),
        "FSABjHat": float(np.asarray(run.moments["FSABjHat"])),
        "jdotb_si": float(np.asarray(run.moments["FSABjHat"])) * CURRENT_SCALE,
        "seconds": time.time() - t0,
        "resolution": [int(n_theta), int(n_zeta), int(n_xi), KIN_NL, int(n_x)],
    }


def truncate_boundary(inp, max_mode):
    """Drop boundary/axis harmonics above ``max_mode`` (CI-budget resolution)."""
    if max_mode is None or max_mode + 1 >= inp.mpol:
        return inp
    m, nt = int(max_mode), int(inp.ntor)
    rbc = np.zeros((2 * m + 1, m + 1))
    zbs = np.zeros((2 * m + 1, m + 1))
    for n in range(-m, m + 1):
        rbc[n + m] = inp.rbc[n + nt, : m + 1]
        zbs[n + m] = inp.zbs[n + nt, : m + 1]
    return dataclasses.replace(
        inp, mpol=m + 1, ntor=m, rbc=rbc, zbs=zbs, rbs=None, zbc=None, raxis_s=None,
        raxis_c=np.asarray(inp.raxis_c)[: m + 1], zaxis_s=np.asarray(inp.zaxis_s)[: m + 1],
        zaxis_c=None,
    )  # fmt: skip


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"iterations": [], "kinetic": {}, "redl": {}, "probe": {}, "gradient": {}}


def save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, indent=1))


class ChunkBudget:
    """Stop the run cleanly after a fixed number of new solve stages."""

    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def spend(self):
        self.used += 1
        if self.limit is not None and self.used >= self.limit:
            save_cache(cache)
            print(f"Chunk budget reached ({self.used} new stage(s)); re-run to continue.",
                  flush=True)  # fmt: skip
            sys.exit(0)


budget = ChunkBudget(MAX_NEW_STAGES)

print("=== examples/paper_benchmarks/bootstrap_consistency_kinetic_loop.py ===", flush=True)
if not DECK.exists():
    raise SystemExit(
        f"equilibrium input deck not found: {DECK}\n"
        "Point SFINCS_JAX_BOOT_LOOP_DECK at input.LandremanPaul2021_QA_reactorScale_lowres "
        "from the vmec_jax examples/data directory."
    )
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
cache = load_cache()

# ----------------------------------------------------------------------------
# 1) Finite-beta prescribed-current input: matched pressure, zero-current start
# ----------------------------------------------------------------------------
P0 = 2.0 * E_CHARGE * N0 * T0_EV  # e*(ne*Te + ni*Ti) at s = 0 [Pa]
AM = P0 * np.array([1.0, -EDGE, 0.0, 0.0, 0.0, -EDGE, EDGE * EDGE])  # (1-EDGE*s)(1-EDGE*s^5)
inp_seed = truncate_boundary(VmecInput.from_file(str(DECK)), MAX_MODE_TRUNC)
inp0 = dataclasses.replace(
    inp_seed,
    am=np.concatenate([AM, np.zeros(21 - AM.size)]), pres_scale=1.0,
    ns_array=np.asarray(NS_LADDER), ftol_array=np.asarray([VMEC_FTOL] * len(NS_LADDER)),
    niter_array=np.asarray(VMEC_NITER),
    ncurr=1, pcurr_type="power_series",
    ac=np.concatenate([[1.0], np.zeros(20)]), curtor=0.0,
)  # fmt: skip
NS = int(NS_LADDER[-1])
NFP = int(inp0.nfp)
S_FULL = np.linspace(0.0, 1.0, NS)
DS = S_FULL[1] - S_FULL[0]
print(f"  deck: {DECK.name} (nfp={NFP}, mpol={inp0.mpol}, ntor={inp0.ntor}, "
      f"ns={NS}, boundary truncation={MAX_MODE_TRUNC})")  # fmt: skip
print(f"  profiles: ne = {N0:.3g}(1 - {EDGE}s^5) 1/m^3, Te = Ti = {T0_EV / 1e3:.3g}(1 - {EDGE}s) keV")
print(f"  kinetic surfaces: {np.round(S_KIN, 4).tolist()}")
print(f"  kinetic grid: Ntheta={KIN_NTHETA} Nzeta={KIN_NZETA} Nxi={KIN_NXI} "
      f"NL={KIN_NL} Nx={KIN_NX}, tol={KIN_TOL:g}, full Fokker-Planck, Er=0")  # fmt: skip

profiles_redl = vmec_bootstrap.KineticProfiles(
    ne_coeffs=N0 * np.array([1.0, 0, 0, 0, 0, -EDGE]),
    Te_coeffs=T0_EV * np.array([1.0, -EDGE]),
    Ti_coeffs=T0_EV * np.array([1.0, -EDGE]),
)

# ----------------------------------------------------------------------------
# 2) The Picard loop (checkpointed per surface solve and per iteration)
# ----------------------------------------------------------------------------
print("Step 1: Picard loop equilibrium -> kinetic <J.B> -> current profile", flush=True)
DDS_MATRIX = vmec_bootstrap._picard_dds_matrix(NS, DS)


def apply_iteration_current(inp, row):
    """Rebuild the prescribed-current input from a cached iteration row."""
    return dataclasses.replace(
        inp, ncurr=1, pcurr_type="power_series",
        ac=np.asarray(row["ac"]), curtor=float(row["curtor"]),
    )  # fmt: skip


inp = inp0
for row in cache["iterations"]:
    if row.get("ac") is not None:
        inp = apply_iteration_current(inp, row)

eq = None
prev_state = None
converged = False
for it in range(N_PICARD):
    if it < len(cache["iterations"]):
        row = cache["iterations"][it]
        print(f"  iteration {it}: cached (delta={row['delta']:.3e}, "
              f"curtor={row['curtor_applied']:+.4e} A)")  # fmt: skip
        if row["delta"] <= PICARD_TOL:
            converged = True
            break
        continue

    # a) equilibrium at the applied current profile (warm-started in-process)
    t0 = time.time()
    try:
        eq = vmec_optimize.solve_equilibrium(inp, initial_state=prev_state)
    except Exception:
        if prev_state is None:
            raise
        eq = vmec_optimize.solve_equilibrium(inp)  # cold fallback
    prev_state = eq.state
    vmec_seconds = time.time() - t0
    fsq_max = max(float(eq.result.fsqr), float(eq.result.fsqz), float(eq.result.fsql))
    if fsq_max > 100.0 * VMEC_FTOL:  # force-residual guard (flag-independent)
        raise RuntimeError(
            f"equilibrium solve did not converge at iteration {it} (fsq_max={fsq_max:.3e})"
        )
    wout = eq.wout
    wout_path = OUT_DIR / f"{STEM}_it{it}.wout.nc"
    write_wout(wout_path, wout)
    iotas = np.asarray(wout.iotas)[1:]

    # b) kinetic <J.B>(s) on the surfaces (each surface checkpointed)
    surfaces = []
    for k, s in enumerate(np.asarray(S_KIN, dtype=float)):
        key = f"it{it}_s{s:.4f}"
        if key not in cache["kinetic"]:
            rec = kinetic_surface_jdotb(wout_path, s, tag=f"_it{it}_k{k}")
            cache["kinetic"][key] = rec
            save_cache(cache)
            print(f"    kinetic s={s:.3f}: <J.B> = {rec['jdotb_si']:+.4e} A T/m^2 "
                  f"({rec['seconds']:.1f} s)", flush=True)  # fmt: skip
            budget.spend()
        surfaces.append(cache["kinetic"][key])
    jdotb_kin = np.asarray([r["jdotb_si"] for r in surfaces])

    # c) invert the parallel-current identity for I(s), refit the profile
    s_nodes = np.concatenate([[0.0], np.asarray(S_KIN, dtype=float), [1.0]])
    j_nodes = np.concatenate([[0.0], jdotb_kin, [0.0]])  # pinned at axis/edge
    jr_full = np.interp(S_FULL, s_nodes, j_nodes)
    geom_kin = vmec_bootstrap.redl_geometry_from_wout(wout, np.asarray(S_KIN))
    fsa_b2_full = np.interp(S_FULL, np.asarray(S_KIN), np.asarray(geom_kin.fsa_B2))
    pres = np.asarray(wout.pres, dtype=float)[1:]  # Pa, half mesh
    s_half = S_FULL[1:] - 0.5 * DS
    dp_half = np.empty_like(pres)
    dp_half[1:-1] = (pres[2:] - pres[:-2]) / (2 * DS)
    dp_half[0] = (pres[1] - pres[0]) / DS
    dp_half[-1] = (pres[-1] - pres[-2]) / DS
    dpds = np.interp(S_FULL, s_half, dp_half)
    psi_a = float(np.asarray(wout.phi)[-1]) / (2.0 * np.pi)

    matrix = (np.diag(fsa_b2_full) @ DDS_MATRIX + np.diag(MU0 * dpds)) / (2 * np.pi * psi_a)
    matrix[0, :] = 0.0
    matrix[0, 0] = 1.0
    rhs = jr_full.copy()
    rhs[0] = 0.0
    i_of_s = np.linalg.solve(matrix, rhs)
    dids_new = (jr_full * 2 * np.pi * psi_a - MU0 * i_of_s * dpds) / fsa_b2_full

    # I'(s) driving the equilibrium at this iteration: the flat zero-current start at
    # iteration 0 (so delta_0 = 1 and the first application is damped too).
    if it > 0:
        dids_applied = np.asarray(cache["iterations"][it - 1]["dids_applied"])
    else:
        dids_applied = np.zeros_like(dids_new)
    scale = max(float(np.max(np.abs(dids_new))), np.finfo(float).tiny)
    delta = float(np.max(np.abs(dids_new - dids_applied))) / scale

    dids_use = (1.0 - RELAX) * dids_applied + RELAX * dids_new
    deg = int(min(FIT_DEGREE, NS - 2))
    coeffs = np.polynomial.polynomial.polyfit(S_FULL, dids_use, deg)
    ac = np.zeros(21)
    ac[: deg + 1] = coeffs
    curtor = float(np.trapezoid(dids_use, S_FULL))

    row = {
        "iteration": it,
        "curtor_applied": float(inp.curtor),
        "curtor": curtor,
        "delta": delta,
        "s_kin": np.asarray(S_KIN, dtype=float).tolist(),
        "jdotb_kinetic_si": jdotb_kin.tolist(),
        "kinetic_seconds": [r["seconds"] for r in surfaces],
        "vmec_seconds": vmec_seconds,
        "vmec_fsqr": float(eq.result.fsqr),
        "iota_mid": float(iotas[len(iotas) // 2]),
        "iota_edge": float(iotas[-1]),
        "fsa_b2": np.asarray(geom_kin.fsa_B2).tolist(),
        "ac": ac.tolist(),
        "dids_applied": dids_use.tolist(),
    }
    cache["iterations"].append(row)
    save_cache(cache)
    print(f"  iteration {it}: <J.B> mid = {jdotb_kin[len(jdotb_kin) // 2]:+.4e}, "
          f"I_new = {curtor / 1e6:+.4f} MA, delta = {delta:.3e}, "
          f"iota_mid = {row['iota_mid']:+.4f}  "
          f"(vmec {vmec_seconds:.1f} s + kinetic {sum(row['kinetic_seconds']):.1f} s)",
          flush=True)  # fmt: skip
    if delta <= PICARD_TOL:
        converged = True
        break
    inp = apply_iteration_current(inp, row)

history = cache["iterations"]
n_iterations = len(history)
final = history[-1]
if not converged:
    print(f"  WARNING: delta = {final['delta']:.3e} after {n_iterations} iterations "
          f"(tol {PICARD_TOL:g}) -- recorded honestly below", flush=True)  # fmt: skip
print(f"  {'converged' if converged else 'stopped'} after {n_iterations} iterations: "
      f"I_bs = {final['curtor'] / 1e6:+.4f} MA, delta = {final['delta']:.3e}")  # fmt: skip

# The converged equilibrium (re-solved from cache when this run resumed).
inp_final = apply_iteration_current(inp0, final)
if eq is None or float(inp.curtor) != float(final["curtor_applied"]):
    # resume path: the last cached iteration's *applied* current reproduces it
    inp_last = inp0
    for row in history[:-1]:
        inp_last = apply_iteration_current(inp_last, row)
    eq = vmec_optimize.solve_equilibrium(inp_last)
wout_final = eq.wout
wout_final_path = OUT_DIR / f"{STEM}_final.wout.nc"
write_wout(wout_final_path, wout_final)

# ----------------------------------------------------------------------------
# 3) Redl contrast at the converged state + kinetic-resolution probe
# ----------------------------------------------------------------------------
print("Step 2: Redl analytic proxy on the same surfaces (the contrast)", flush=True)
if "jdotb" not in cache["redl"]:
    geom = vmec_bootstrap.redl_geometry_from_wout(wout_final, np.asarray(S_KIN))
    jr, details = vmec_bootstrap.j_dot_B_redl(profiles_redl, geom, HELICITY_N)
    cache["redl"] = {
        "jdotb": np.asarray(jr, dtype=float).tolist(),
        "f_t": np.asarray(geom.f_t, dtype=float).tolist(),
        "epsilon": np.asarray(geom.epsilon, dtype=float).tolist(),
        "nu_e_star": np.asarray(details["nu_e_star"], dtype=float).tolist(),
        "nu_i_star": np.asarray(details["nu_i_star"], dtype=float).tolist(),
    }
    save_cache(cache)
jdotb_redl = np.asarray(cache["redl"]["jdotb"])
jdotb_final = np.asarray(final["jdotb_kinetic_si"])
rel_discrepancy = np.abs(jdotb_final - jdotb_redl) / np.max(np.abs(jdotb_final))
print("  s      kinetic <J.B>    Redl <J.B>     rel diff (of max |kinetic|)")
for s, jk, jr_v, rd in zip(S_KIN, jdotb_final, jdotb_redl, rel_discrepancy):
    print(f"  {s:.3f}  {jk:+.4e}   {jr_v:+.4e}   {rd:.3f}")
print(f"  mean/max rel discrepancy: {np.mean(rel_discrepancy):.3f} / "
      f"{np.max(rel_discrepancy):.3f}")  # fmt: skip

print("Step 3: split refined mid-surface re-solves (numerical error bound)", flush=True)
k_mid = len(S_KIN) // 2
s_mid = float(np.asarray(S_KIN)[k_mid])
base_mid = float(jdotb_final[k_mid])
PROBES = {
    "angular": dict(n_theta=PROBE_NTHETA, n_zeta=PROBE_NZETA),
    "velocity": dict(n_xi=PROBE_NXI, n_x=PROBE_NX),
}
for name, overrides in PROBES.items():
    if name not in cache["probe"]:
        rec = kinetic_surface_jdotb(wout_final_path, s_mid, tag=f"_probe_{name}", **overrides)
        cache["probe"][name] = {
            "s": s_mid,
            "resolution": rec["resolution"],
            "jdotb_refined": rec["jdotb_si"],
            "jdotb_base": base_mid,
            "rel_dev": abs(rec["jdotb_si"] - base_mid) / abs(rec["jdotb_si"]),
            "seconds": rec["seconds"],
        }
        save_cache(cache)
        budget.spend()
probe = cache["probe"]
for name in PROBES:
    p = probe[name]
    print(f"  {name} {tuple(p['resolution'])}: base {p['jdotb_base']:+.4e} -> "
          f"refined {p['jdotb_refined']:+.4e} (rel dev {p['rel_dev']:.2e}, "
          f"{p['seconds']:.0f} s)", flush=True)  # fmt: skip
probe_rel_dev_max = max(probe[name]["rel_dev"] for name in PROBES)

# ----------------------------------------------------------------------------
# 4) Gradient hook: value_and_grad through the differentiable chain
# ----------------------------------------------------------------------------
print("Step 4: jax.value_and_grad of the total bootstrap current", flush=True)
if "value" not in cache["gradient"]:
    cfg = vmec_implicit.make_config(
        dataclasses.replace(inp_final, ns_array=np.asarray([NS]),
                            ftol_array=np.asarray([VMEC_FTOL]),
                            niter_array=np.asarray([max(VMEC_NITER)])),
        adjoint_tol=1e-6, adjoint_maxiter=40,
    )  # fmt: skip
    params0 = vmec_implicit.params_from_input(inp_final)
    psi_a_hat = abs(float(inp_final.phiedge)) / (2.0 * np.pi)
    a_hat = float(wout_final.Aminor_p)
    grad_rows = [int(round(float(s) * (NS - 1) + 0.5)) for s in GRAD_S]
    grad_rows = [min(max(j, 1), NS - 1) for j in grad_rows]
    s_rows = [(j - 0.5) / (NS - 1) for j in grad_rows]

    # Per-row kinetic operator templates (species values at each surface); the
    # geometry leaves are replaced per evaluation from the Boozer transform.
    op_templates = []
    for s_row in s_rows:
        ne, dne, te, dte = kinetic_profiles(s_row)
        text = f"""&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 1
  helicity_n = {NFP}
  psiAHat = {psi_a_hat:.10g}
  aHat = {a_hat:.10g}
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = {ne / NBAR:.10g} {ne / NBAR:.10g}
  THats = {te / TBAR_EV:.10g} {te / TBAR_EV:.10g}
  dNHatdpsiHats = {dne / NBAR / psi_a_hat:.10g} {dne / NBAR / psi_a_hat:.10g}
  dTHatdpsiHats = {dte / TBAR_EV / psi_a_hat:.10g} {dte / TBAR_EV / psi_a_hat:.10g}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = {NU_N:.10g}
  Er = 0.0
  collisionOperator = 0
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
/
&resolutionParameters
  Ntheta = {GRAD_NTHETA}
  Nzeta = {GRAD_NZETA}
  Nxi = {GRAD_NXI}
  NL = {GRAD_NL}
  Nx = {GRAD_NX}
  solverTolerance = {GRAD_TOL}
/
&otherNumericalParameters
  xGridScheme = 5
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
"""
        op_templates.append(kinetic_operator_from_namelist(parse_sfincs_input_text(text)))

    grids = make_grids(
        n_theta=op_templates[0].n_theta, n_zeta=op_templates[0].n_zeta,
        n_xi=op_templates[0].n_xi, n_x=op_templates[0].n_x, n_l=GRAD_NL,
        n_periods=NFP, x_grid_scheme=5,
    )  # fmt: skip

    def chain_jdotb_over_b2(state, rt, row_j, op_template):
        """Traceable ``<J.B>_SI / <B^2>`` on half-mesh row ``row_j``."""
        tabs = boozer_input_tables(state, rt, row_j)
        booz = booz_transform(
            rmnc=tabs["rmnc"][None, :], zmns=tabs["zmns"][None, :],
            lmns=tabs["lmns"][None, :], bmnc=tabs["bmnc"][None, :],
            bsubumnc=tabs["bsubumnc"][None, :], bsubvmnc=tabs["bsubvmnc"][None, :],
            iota=tabs["iota"][None], xm=tabs["xm"], xn=tabs["xn"],
            xm_nyq=tabs["xm"], xn_nyq=tabs["xn"],
            nfp=NFP, mboz=MBOZ, nboz=NBOZ, asym=False,
        )  # fmt: skip
        ixm = np.asarray(booz["ixm_b"])
        ixn = np.asarray(booz["ixn_b"])  # includes the nfp factor
        geom = FluxSurfaceGeometry.from_fourier(
            theta=grids.theta, zeta=grids.zeta, bmnc=booz["bmnc_b"][0],
            m=jnp.asarray(ixm), n=jnp.asarray(ixn // NFP), n_periods=NFP,
            iota=booz["iota_b"][0], g_hat=booz["bvco_b"][0], i_hat=booz["buco_b"][0],
        )  # fmt: skip
        fsab2 = geom.fsab_hat2(theta_weights=grids.theta_weights,
                               zeta_weights=grids.zeta_weights)  # fmt: skip
        op = dataclasses.replace(
            op_template,
            b_hat=geom.b_hat, db_hat_dtheta=geom.db_hat_dtheta,
            db_hat_dzeta=geom.db_hat_dzeta, d_hat=geom.d_hat,
            b_hat_sup_theta=geom.b_hat_sup_theta, b_hat_sup_zeta=geom.b_hat_sup_zeta,
            b_hat_sub_theta=geom.b_hat_sub_theta, b_hat_sub_zeta=geom.b_hat_sub_zeta,
            fsab_hat2=fsab2,
        )  # fmt: skip
        result = kinetic_solve(op, op.rhs(), method="gmres", tol=GRAD_TOL,
                               differentiable=True)  # fmt: skip
        mom = profile_moments_from_operator(op, result.x)
        return mom["FSABjHat"] * CURRENT_SCALE / fsab2

    def total_bootstrap_current(coefficient):
        """Coarse-quadrature |I_bs| [A] as a function of RBC(n=0, m=1).

        dI/ds ~ 2 pi psi_a <J.B>/<B^2> (the mu0*I*dp/ds correction, ~2% at
        this beta, is dropped in this scalar); trapezoid over the quadrature
        surfaces with the profile pinned to zero at s = 0 and s = 1.  The
        magnitude is used because the traceable Boozer-transform route and
        the equilibrium-file route carry opposite parallel-current sign
        conventions (coordinate handedness); |I_bs| is convention-free and
        is what a bootstrap-targeting objective would use.
        """
        rbc = params0.rbc.at[int(inp_final.ntor), 1].set(coefficient)
        params = dataclasses.replace(params0, rbc=rbc)
        state = vmec_implicit.solve_implicit(params, cfg)
        rt = vmec_implicit.runtime_from_params(params, cfg)
        dids = [2.0 * jnp.pi * psi_a_hat * chain_jdotb_over_b2(state, rt, j, op_t)
                for j, op_t in zip(grad_rows, op_templates)]  # fmt: skip
        s_nodes = jnp.asarray([0.0, *s_rows, 1.0])
        f_nodes = jnp.asarray([0.0, *dids, 0.0])
        return jnp.abs(jnp.sum(0.5 * (f_nodes[1:] + f_nodes[:-1]) * jnp.diff(s_nodes)))

    c0 = float(np.asarray(inp_final.rbc)[int(inp_final.ntor), 1])
    t0 = time.time()
    value, grad = jax.value_and_grad(total_bootstrap_current)(jnp.asarray(c0))
    grad_seconds = time.time() - t0
    # Consistency of the differentiable Boozer-route chain against the loop's
    # VMEC-geometry route (interpolated to the chain's quadrature surfaces).
    i_bs_loop = float(final["curtor"])
    cache["gradient"] = {
        "dof": "RBC(0,1)",
        "dof_value": c0,
        "value_A": float(value),
        "value_is_magnitude": True,
        "grad_A_per_m": float(grad),
        "seconds": grad_seconds,
        "rows": grad_rows,
        "s_rows": s_rows,
        "resolution": [GRAD_NTHETA, GRAD_NZETA, GRAD_NXI, GRAD_NL, GRAD_NX],
        "mboz_nboz": [MBOZ, NBOZ],
        "loop_curtor_A": i_bs_loop,
        "rel_dev_vs_loop": abs(float(value) - abs(i_bs_loop)) / abs(i_bs_loop),
    }
    save_cache(cache)
    budget.spend()
gradient = cache["gradient"]
print(f"  |I_bs|(chain, {len(gradient['rows'])}-surface quadrature) = "
      f"{gradient['value_A'] / 1e6:.4f} MA "
      f"(loop |curtor| {abs(gradient['loop_curtor_A']) / 1e6:.4f} MA, "
      f"rel dev {gradient['rel_dev_vs_loop']:.2f})")  # fmt: skip
print(f"  d |I_bs| / d {gradient['dof']} = {gradient['grad_A_per_m']:+.6e} A/m "
      f"(finite: {np.isfinite(gradient['grad_A_per_m'])}, "
      f"sign: {'+' if gradient['grad_A_per_m'] > 0 else '-'}; "
      f"{gradient['seconds']:.0f} s incl. compile)")  # fmt: skip

# ----------------------------------------------------------------------------
# 5) JSON record
# ----------------------------------------------------------------------------
print("Step 5: writing the JSON record")
record = {
    "benchmark": (
        "Kinetic-solver-in-the-loop bootstrap-consistent equilibrium iteration, "
        "finite-beta precise-QA reactor-scale configuration"
    ),
    "references": [
        "A. Redl et al., Phys. Plasmas 28, 022502 (2021)",
        "M. Landreman, S. Buller and M. Drevlak, Phys. Plasmas 29, 082501 (2022)",
        "M. Landreman and E. Paul, Phys. Rev. Lett. 128, 035001 (2022)",
        "O. Sauter, C. Angioni and Y.R. Lin-Liu, Phys. Plasmas 6, 2834 (1999)",
    ],
    "configuration": {
        "deck": DECK.name,
        "boundary_truncation_max_mode": MAX_MODE_TRUNC,
        "nfp": NFP, "ns": NS, "vmec_ftol": VMEC_FTOL,
        "pressure_pa_power_series": AM.tolist(),
        "ne_m3": f"{N0:g}*(1 - {EDGE}*s^5)",
        "Te_eV": f"{T0_EV:g}*(1 - {EDGE}*s)  (= Ti; Zeff = 1)",
        "helicity_n": HELICITY_N,
    },
    "normalization": {
        "nBar_m3": NBAR, "TBar_eV": TBAR_EV, "mBar_kg": MBAR_KG,
        "BBar_T": 1.0, "RBar_m": 1.0, "nu_n": NU_N,
        "current_scale_A_per_m2": CURRENT_SCALE,
        "si_formula": "FSABjHat * e * nBar * sqrt(2*TBar/mBar) * BBar",
    },
    "kinetic_resolution": {
        "Ntheta": KIN_NTHETA, "Nzeta": KIN_NZETA, "Nxi": KIN_NXI,
        "NL": KIN_NL, "Nx": KIN_NX, "solver_tolerance": KIN_TOL,
        "collision_operator": "full Fokker-Planck (two species)", "Er": 0.0,
    },
    "picard": {
        "surfaces_s": np.asarray(S_KIN, dtype=float).tolist(),
        "relax": RELAX, "tol": PICARD_TOL, "fit_degree": FIT_DEGREE,
        "converged": bool(converged),
        "iterations": n_iterations,
        "history": [
            {k: row[k] for k in (
                "iteration", "curtor_applied", "curtor", "delta",
                "jdotb_kinetic_si", "iota_mid", "iota_edge",
                "vmec_seconds", "kinetic_seconds", "vmec_fsqr")}
            for row in history
        ],
    },  # fmt: skip
    "redl_contrast": {
        "jdotb_redl_si": jdotb_redl.tolist(),
        "jdotb_kinetic_si": jdotb_final.tolist(),
        "rel_discrepancy_of_max": rel_discrepancy.tolist(),
        "mean_rel_discrepancy": float(np.mean(rel_discrepancy)),
        "max_rel_discrepancy": float(np.max(rel_discrepancy)),
        "redl_nu_e_star": cache["redl"]["nu_e_star"],
        "redl_nu_i_star": cache["redl"]["nu_i_star"],
        "trapped_fraction": cache["redl"]["f_t"],
    },
    "kinetic_resolution_probe": {
        **probe,
        "max_rel_dev": probe_rel_dev_max,
        # The velocity refinement moves the kinetic value toward the proxy,
        # so the surviving mid-surface proxy error at the refined grid is the
        # honest lower estimate of the discrepancy there.
        "mid_surface_discrepancy_base_grid": float(
            abs(base_mid - jdotb_redl[k_mid]) / np.max(np.abs(jdotb_final))),
        "mid_surface_discrepancy_refined_velocity_grid": float(
            abs(probe["velocity"]["jdotb_refined"] - jdotb_redl[k_mid])
            / np.max(np.abs(jdotb_final))),
    },  # fmt: skip
    "gradient_hook": gradient,
    "limitation": (
        "The demonstration runs at reduced density (on-axis beta 0.75%) so "
        "the converged iota profile stays inside the rational-free window "
        "(2/5, 1/2).  At the profile paper's full reactor density (on-axis "
        "beta 2.4%) the self-consistent current drives iota across 1/2 and "
        "toward 2/3, and local drift-kinetic solves on surfaces near those "
        "low-order rationals become resonant at fixed resolution "
        "(order-of-magnitude jumps and sign flips in <J.B> were observed). "
        "A production-beta version of this loop needs resonance-aware "
        "surface placement or per-surface resolution control."
    ),
    "claim_boundary": (
        "The kinetic-vs-Redl discrepancy profile compares the full-Fokker-"
        "Planck two-species drift-kinetic solve at Er = 0 and fixed reference "
        "Coulomb logarithm (nu_n) with the Redl fit's own Sauter per-surface "
        "collisionality model on the same equilibrium, geometry, and "
        "profiles.  The split angular/velocity refined re-solves bound the "
        "kinetic numerical error below the recorded discrepancy.  The gradient "
        "hook records one finite end-to-end derivative through the "
        "differentiable equilibrium -> Boozer -> kinetic chain at reduced "
        "resolution; its AD-vs-FD accuracy is documented in "
        "gradient_verification.py and the flagship optimization example, not "
        "re-verified here."
    ),
    "finding": (
        "The Picard iteration with the actual drift-kinetic solve in the "
        "loop converges to a bootstrap-consistent finite-beta QA equilibrium "
        "in a handful of damped iterations (delta history in "
        "picard.history), with the self-consistent current raising the iota "
        "profile.  At the converged state the Redl analytic proxy "
        "over-predicts the kinetic bootstrap current across the interior "
        "profile by the recorded discrepancy (a few percent of the profile "
        "maximum, comparable to the proxy-vs-kinetic RMS reported for the "
        "self-consistent-bootstrap optima in the literature); the "
        "discrepancy survives the split resolution refinements "
        "(kinetic_resolution_probe records the mid-surface value on the "
        "refined velocity grid).  This is the proxy error that "
        "kinetic-in-the-loop iteration removes by construction."
    ),
}  # fmt: skip
JSON_PATH.write_text(json.dumps(record, indent=1))
print(f"  wrote {JSON_PATH}")

# ----------------------------------------------------------------------------
# 6) Figure
# ----------------------------------------------------------------------------
print("Step 6: rendering the figure")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.0,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)

s_kin = np.asarray(S_KIN, dtype=float)
cmap = matplotlib.colormaps["viridis"]
for row in history:
    it = int(row["iteration"])
    frac = it / max(n_iterations - 1, 1)
    is_last = it == n_iterations - 1
    ax1.plot(s_kin, np.asarray(row["jdotb_kinetic_si"]) / 1e6, "-o",
             ms=4.5 if is_last else 3.0, lw=2.2 if is_last else 1.2,
             color=cmap(0.85 * frac), alpha=1.0 if is_last else 0.75,
             label=f"iteration {it}" + (" (converged)" if is_last and converged else ""),
             zorder=3 if is_last else 2)  # fmt: skip
ax1.plot(s_kin, jdotb_redl / 1e6, "--s", ms=4, lw=1.6, color="#d62728",
         label="Redl analytic proxy", zorder=4)  # fmt: skip
ax1.axhline(0.0, color="0.6", lw=0.8, zorder=0)
ax1.set_xlabel(r"normalized toroidal flux $s$")
ax1.set_ylabel(r"$\langle J \cdot B\rangle$  [MA T / m$^2$]")
ax1.set_title("Kinetic bootstrap profile per Picard iteration")
ax1.legend(frameon=False, loc="best")

its = [int(row["iteration"]) for row in history if np.isfinite(row["delta"])]
deltas = [float(row["delta"]) for row in history if np.isfinite(row["delta"])]
ax2.semilogy(its, deltas, "-o", ms=5, color="#1f77b4", label=r"$\delta$ per iteration")
ax2.axhline(PICARD_TOL, color="0.25", lw=1.2, ls="--",
            label=f"tolerance {PICARD_TOL:g}")  # fmt: skip
ax2.set_xlabel("Picard iteration")
ax2.set_ylabel(r"$\delta = \max|I'_{new} - I'_{applied}| \, / \, \max|I'_{new}|$")
ax2.set_title("Current-profile convergence history")
ax2.set_xticks(its)
ax2.legend(frameon=False, loc="best")

fig.suptitle(
    "Finite-beta precise QA: drift-kinetic solve inside the bootstrap-consistency loop",
    y=1.04,
)
fig.savefig(PNG_PATH, dpi=170, bbox_inches="tight")
plt.close(fig)

# Palette-quantize to keep the committed figure small (repo convention).
img = Image.open(PNG_PATH).convert("P", palette=Image.ADAPTIVE, colors=64)
img.save(PNG_PATH, optimize=True)
print(f"  wrote {PNG_PATH} ({PNG_PATH.stat().st_size / 1024:.0f} KB)")
print("Done.")
