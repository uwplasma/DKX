"""Classical and neoclassical impurity transport: a mixed-collisionality benchmark.

Roadmap item 4 of ``plan_final.md``.  For a bulk hydrogenic plasma plus a
high-Z trace impurity this script assembles the classical and neoclassical
impurity-transport diagnostics:

  1. Fortran parity anchor -- the committed two-species (H + fully-stripped
     carbon, Z = 6) SFINCS v3 example ``quick_2species_FPCollisions_noEr``:
     dkx reproduces the frozen Fortran state, so its *neoclassical* impurity
     particle flux ``particleFlux_vm_psiHat`` is checked directly against the
     Fortran ``sfincsOutput.h5`` golden.

  2. Mixed-collisionality scan on the W7-X standard configuration (Boozer
     ``.bc``, r/a = 0.5, real ``|grad psiHat|^2`` metric).  At each
     collisionality ``nu_n`` a full multi-species RHSMode=1 kinetic solve gives
     the neoclassical impurity flux, and the local classical flux is the
     algebraic :func:`dkx.impurity.classical_species_fluxes`.  The classical
     flux scales linearly with ``nu_n`` (collisional); the neoclassical flux
     approaches that scaling at high collisionality (Pfirsch-Schlueter) and
     departs from it toward lower collisionality.  At the first point the
     classical flux from ``dkx.impurity`` is checked against dkx's own
     ``classicalParticleFlux_psiHat`` (the ``classicalTransport.F90`` counterpart)
     to solver-independent machine precision -- the algebraic-limit anchor.

  3. Charge-state scan -- :func:`dkx.impurity.classical_impurity_flux_over_charge_states`
     maps the classical impurity flux over a sweep of ``Z`` in one batched
     (``vmap``) call, showing the Z-scaling of impurity accumulation.

  4. Temperature screening -- the ion-density peaking coefficient (exactly
     ``-Z``), the ion-temperature screening coefficient (``-> 1/2`` in the
     collisional heavy-impurity limit), and an AD-vs-FD check of
     ``d Gamma_z / d(dT_i/dpsiHat)`` (the screening-aware gradient).

Primary literature: Braginskii, Rev. Plasma Phys. 1, 205 (1965); Rutherford,
Phys. Fluids 17, 1782 (1974); Hinton & Hazeltine, Rev. Mod. Phys. 48, 239
(1976); Wenzel & Sigmar, Nucl. Fusion 30, 1117 (1990); Helander & Sigmar,
*Collisional Transport in Magnetized Plasmas*, CUP (2002).

Expected runtime: ~2-3 min on a laptop CPU (one tiny Fortran-parity solve plus
a five-point W7-X kinetic scan at modest resolution; the classical Z-scan and
screening pieces are algebraic and instant).  Finished scan points are cached
in ``output/`` and skipped on re-runs.

Run (from the repo root):
  DKX_EQUILIBRIA_DIRS=/path/to/equilibria python examples/paper_benchmarks/impurity_transport.py
"""

import json
import lzma
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dkx import impurity as imp
from dkx.io import read_sfincs_h5
from dkx.moments import FluxSurface
from dkx.run import run_profile
from dkx.species import SpeciesSet

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
EQUILIBRIUM = "w7x_standardConfig.bc"  # resolved via DKX_EQUILIBRIA_DIRS / data cache
RN_WISH = 0.5  # benchmark surface r/a = 0.5

# Bulk hydrogen + trace fully-stripped carbon (Z = 6), both peaked.
BULK_Z, BULK_M, BULK_N, BULK_T = 1.0, 1.0, 0.6, 1.0
BULK_DN_DR, BULK_DT_DR = -0.6, -1.0  # d/drHat (log-gradient ~ -1 for both)
IMP_Z, IMP_M, IMP_N, IMP_T = 6.0, 12.0, 0.006, 1.0
IMP_DN_DR, IMP_DT_DR = -0.006, -1.0  # matched impurity density log-gradient

# Collisionality scan (the v3 scalar nu_n): moderate -> high, five points.
# Capped where the multi-species Fokker-Planck Krylov solve stays robust at
# this modest resolution (the highest points span the Pfirsch-Schlueter branch).
NU_N_SCAN = [8.4774e-3, 2.4e-2, 6.8e-2, 1.6e-1, 3.0e-1]

# Modest kinetic resolution (a decomposition/scaling demo, not a converged
# transport number -- recorded in the JSON provenance).  Kept below the
# tier-3 dense-fallback size so every point solves even if the Krylov tier
# is exhausted.
N_THETA, N_ZETA, N_XI, N_X = 7, 13, 8, 5
SOLVER_TOLERANCE = 1e-8
DELTA = 4.5694e-3

# Charge-state scan (fully-stripped, A ~ 2Z) at the reference collisionality.
Z_SCAN = [2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0]
Z_SCAN_NU_N = 8.4774e-3
Z_SCAN_N_HAT = 1e-4  # small enough to keep the heaviest charge state trace-like

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "output"
CACHE_PATH = OUT_DIR / "impurity_transport_cache.json"
FIG_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks"
JSON_PATH = FIG_DIR / "impurity_transport.json"
PNG_PATH = FIG_DIR / "impurity_transport.png"

# The committed two-species carbon Fortran fixture (bulk H + Z=6 carbon).
# The heavy Fortran output golden is committed only in lzma-compressed form
# (`*.h5.xz`); decompress it into `output/` on demand (see _fortran_golden).
FORTRAN_DECK = REPO_ROOT / "tests" / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
FORTRAN_GOLDEN_XZ = REPO_ROOT / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5.xz"

DECK_TEMPLATE = """! W7-X standard configuration, classical/neoclassical impurity benchmark deck.
! Generated by examples/paper_benchmarks/impurity_transport.py
&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 11
  equilibriumFile = "{equilibrium}"
  inputRadialCoordinate = 3
  rN_wish = {rn_wish}
/
&speciesParameters
  Zs = {bulk_z} {imp_z}
  mHats = {bulk_m} {imp_m}
  nHats = {bulk_n} {imp_n}
  THats = {bulk_t} {imp_t}
  dNHatdrHats = {bulk_dn} {imp_dn}
  dTHatdrHats = {bulk_dt} {imp_dt}
/
&physicsParameters
  Delta = {delta}
  alpha = 1.0d+0
  nu_n = {nu_n}
  Er = 0
  collisionOperator = 0
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
  useDKESExBDrift = .false.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {n_theta}
  Nzeta = {n_zeta}
  Nxi = {n_xi}
  Nx = {n_x}
  solverTolerance = {tol}
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
"""


def _w7x_deck(nu_n: float, out_dir: Path) -> Path:
    text = DECK_TEMPLATE.format(
        equilibrium=EQUILIBRIUM, rn_wish=RN_WISH,
        bulk_z=BULK_Z, imp_z=IMP_Z, bulk_m=BULK_M, imp_m=IMP_M,
        bulk_n=BULK_N, imp_n=IMP_N, bulk_t=BULK_T, imp_t=IMP_T,
        bulk_dn=BULK_DN_DR, imp_dn=IMP_DN_DR, bulk_dt=BULK_DT_DR, imp_dt=IMP_DT_DR,
        delta=DELTA, nu_n=nu_n, n_theta=N_THETA, n_zeta=N_ZETA, n_xi=N_XI, n_x=N_X,
        tol=SOLVER_TOLERANCE,
    )  # fmt: skip
    path = out_dir / f"w7x_impurity_nu_{nu_n:.6e}.input.namelist"
    path.write_text(text)
    return path


def _fortran_golden(out_dir: Path) -> Path:
    """Return a readable Fortran golden ``.h5``, decompressing the ``.xz`` if needed."""
    uncompressed = FORTRAN_GOLDEN_XZ.with_suffix("")  # strip .xz -> .h5
    if uncompressed.exists():
        return uncompressed
    target = out_dir / uncompressed.name
    if not target.exists():
        target.write_bytes(lzma.open(FORTRAN_GOLDEN_XZ).read())
    return target


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"fortran": None, "scan": {}}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=1))


def _geometry_factor_from_run(run, h5_path: Path) -> float:
    """Flux-surface geometry scalar ``G = <|grad psiHat|^2/BHat^2>`` from a run.

    ``gpsiHatpsiHat`` is written by the output writer in Fortran (zeta, theta)
    layout; transpose to the operator's (theta, zeta) layout and average with
    :func:`dkx.impurity.classical_geometry_factor`.
    """
    surf = FluxSurface.from_operator(run.operator)
    gpsipsi_zt = np.asarray(read_sfincs_h5(h5_path)["gpsiHatpsiHat"], dtype=np.float64)
    gpsipsi_tz = jnp.asarray(gpsipsi_zt.T)
    return float(
        imp.classical_geometry_factor(
            theta_weights=surf.theta_weights, zeta_weights=surf.zeta_weights,
            d_hat=surf.d_hat, b_hat=surf.b_hat, gpsipsi=gpsipsi_tz,
        )  # fmt: skip
    )


# ----------------------------------------------------------------------------
# 1) Fortran parity anchor (neoclassical impurity flux, H + carbon)
# ----------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
cache = _load_cache()

print("Step 1: Fortran parity anchor (quick_2species_FPCollisions_noEr, H + C6+)")
if cache["fortran"] is None:
    run = run_profile(FORTRAN_DECK, out_path=None, emit=None)
    golden = read_sfincs_h5(_fortran_golden(OUT_DIR))
    dkx_imp = float(np.asarray(run.moments["particleFlux_vm_psiHat"]).ravel()[1])
    fort_imp = float(np.asarray(golden["particleFlux_vm_psiHat"]).ravel()[1])
    dkx_ion = float(np.asarray(run.moments["particleFlux_vm_psiHat"]).ravel()[0])
    fort_ion = float(np.asarray(golden["particleFlux_vm_psiHat"]).ravel()[0])
    cache["fortran"] = {
        "impurity_neoclassical_flux_dkx": dkx_imp,
        "impurity_neoclassical_flux_fortran": fort_imp,
        "impurity_relative_agreement": abs(dkx_imp - fort_imp) / abs(fort_imp),
        "ion_neoclassical_flux_dkx": dkx_ion,
        "ion_neoclassical_flux_fortran": fort_ion,
        "ion_relative_agreement": abs(dkx_ion - fort_ion) / abs(fort_ion),
        "solver_tolerance": float(run.input.resolution.solver_tolerance),
    }
    _save_cache(cache)
fort = cache["fortran"]
print(f"  impurity neoclassical flux  dkx = {fort['impurity_neoclassical_flux_dkx']:.6e}")
print(f"                          Fortran = {fort['impurity_neoclassical_flux_fortran']:.6e}")
print(f"  relative agreement = {fort['impurity_relative_agreement']:.2e}"
      f" (solver tol {fort['solver_tolerance']:.0e})")  # fmt: skip

# ----------------------------------------------------------------------------
# 2) Mixed-collisionality scan on W7-X (checkpointed)
# ----------------------------------------------------------------------------
print(f"\nStep 2: W7-X mixed-collisionality scan ({len(NU_N_SCAN)} points)")
geometry_factor = None
species_set = None
for nu_n in NU_N_SCAN:
    key = f"{nu_n:.6e}"
    if key not in cache["scan"]:
        t0 = time.perf_counter()
        deck = _w7x_deck(nu_n, OUT_DIR)
        h5_path = OUT_DIR / f"w7x_impurity_nu_{nu_n:.6e}.h5"
        run = run_profile(deck, out_path=h5_path, emit=None)
        neo = float(np.asarray(run.moments["particleFlux_vm_psiHat"]).ravel()[1])
        classical_dkx = float(np.asarray(run.moments["classicalParticleFlux_psiHat"]).ravel()[1])
        neo_ion = float(np.asarray(run.moments["particleFlux_vm_psiHat"]).ravel()[0])
        g = _geometry_factor_from_run(run, h5_path)
        op = run.operator
        species = SpeciesSet(
            z=jnp.asarray(op.z_s), m_hat=jnp.asarray(op.m_hat), n_hat=jnp.asarray(op.n_hat),
            t_hat=jnp.asarray(op.t_hat), dn_hat_dpsi_hat=jnp.asarray(op.dn_hat_dpsi_hat),
            dt_hat_dpsi_hat=jnp.asarray(op.dt_hat_dpsi_hat),
        )  # fmt: skip
        res = imp.classical_impurity_flux(species, geometry_factor=g, delta=float(op.delta), nu_n=nu_n)
        classical_module = float(res.particle_flux)
        cache["scan"][key] = {
            "nu_n": nu_n,
            "impurity_neoclassical_flux": neo,
            "ion_neoclassical_flux": neo_ion,
            "impurity_classical_flux_dkx": classical_dkx,
            "impurity_classical_flux_module": classical_module,
            "impurity_classical_diffusion": float(res.diffusion_coefficient),
            "impurity_strength": float(res.impurity_strength),
            "geometry_factor": g,
            "module_vs_dkx_rel_err": abs(classical_module - classical_dkx) / abs(classical_dkx),
            "converged": bool(run.solve_result.converged),
            "seconds": time.perf_counter() - t0,
        }
        _save_cache(cache)
        print(f"  nu_n = {nu_n:.3e}: neo = {neo:+.3e}, classical = {classical_module:+.3e}"
              f"  ({cache['scan'][key]['seconds']:.1f} s)")  # fmt: skip
    rec = cache["scan"][key]
    if geometry_factor is None:  # keep the geometry/species for steps 3-4
        geometry_factor = rec["geometry_factor"]

# Reference geometry factor + species for the algebraic parts (from the first point).
geometry_factor = cache["scan"][f"{NU_N_SCAN[0]:.6e}"]["geometry_factor"]
consistency = max(cache["scan"][f"{n:.6e}"]["module_vs_dkx_rel_err"] for n in NU_N_SCAN)
print(f"  classical flux: dkx.impurity vs dkx.moments max rel err = {consistency:.2e}")

# ----------------------------------------------------------------------------
# 3) Charge-state scan (vmap over Z) -- classical, algebraic
# ----------------------------------------------------------------------------
print(f"\nStep 3: charge-state scan (Z = {Z_SCAN})")
bulk = SpeciesSet(
    z=jnp.asarray([BULK_Z]), m_hat=jnp.asarray([BULK_M]), n_hat=jnp.asarray([BULK_N]),
    t_hat=jnp.asarray([BULK_T]),
    dn_hat_dpsi_hat=jnp.asarray([BULK_DN_DR * 1.0]),  # gradient sign only; magnitude cancels in log
    dt_hat_dpsi_hat=jnp.asarray([BULK_DT_DR * 1.0]),
)
z_arr = jnp.asarray(Z_SCAN)
m_arr = 2.0 * z_arr
pf_z, hf_z, dcl_z = imp.classical_impurity_flux_over_charge_states(
    bulk, impurity_charges=z_arr, impurity_masses=m_arr, impurity_n_hat=Z_SCAN_N_HAT,
    geometry_factor=geometry_factor, delta=DELTA, nu_n=Z_SCAN_NU_N,
    match_bulk_logarithmic_gradients=True,
)  # fmt: skip
pf_z = np.asarray(pf_z)
screening_coeffs = []
for z_imp in Z_SCAN:
    sp = imp.build_impurity_plasma(
        bulk, impurity_z=z_imp, impurity_m_hat=2.0 * z_imp, impurity_n_hat=Z_SCAN_N_HAT,
        match_bulk_logarithmic_gradients=True,
    )  # fmt: skip
    scr = imp.temperature_screening_diagnostic(sp, geometry_factor=geometry_factor, delta=DELTA, nu_n=Z_SCAN_NU_N)
    screening_coeffs.append(float(scr.screening_coefficient))
for z_imp, gamma, h in zip(Z_SCAN, pf_z, screening_coeffs):
    print(f"  Z = {z_imp:4.0f}: classical flux = {gamma:+.3e}, screening coeff H = {h:.4f}")

# ----------------------------------------------------------------------------
# 4) Temperature screening + AD-vs-FD gradient (carbon, the reference impurity)
# ----------------------------------------------------------------------------
print("\nStep 4: temperature screening and AD-vs-FD gradient (Z = 6 carbon)")
carbon = imp.build_impurity_plasma(
    bulk, impurity_z=IMP_Z, impurity_m_hat=IMP_M, impurity_n_hat=IMP_N,
    match_bulk_logarithmic_gradients=True,
)
scr = imp.temperature_screening_diagnostic(carbon, geometry_factor=geometry_factor, delta=DELTA, nu_n=Z_SCAN_NU_N)


def _gamma_of_dti(dti: jnp.ndarray) -> jnp.ndarray:
    dt = carbon.dt_hat_dpsi_hat.at[0].set(dti)
    pf, _ = imp.classical_species_fluxes(
        z=carbon.z, m_hat=carbon.m_hat, n_hat=carbon.n_hat, t_hat=carbon.t_hat,
        dn_hat_dpsi_hat=carbon.dn_hat_dpsi_hat, dt_hat_dpsi_hat=dt,
        geometry_factor=geometry_factor, delta=DELTA, nu_n=Z_SCAN_NU_N,
    )  # fmt: skip
    return pf[-1]


dti0 = float(carbon.dt_hat_dpsi_hat[0])
ad = float(jax.grad(_gamma_of_dti)(dti0))
h_fd = 1e-6 * max(1.0, abs(dti0))
fd = float((_gamma_of_dti(dti0 + h_fd) - _gamma_of_dti(dti0 - h_fd)) / (2.0 * h_fd))
ad_fd_rel = abs(ad - fd) / max(abs(ad), 1e-300)
print(f"  ion-density peaking coefficient = {float(scr.ion_density_peaking_coefficient):.3f} (exact -Z = {-IMP_Z:.0f})")
print(f"  ion-temperature screening coefficient H = {float(scr.screening_coefficient):.4f} (collisional limit 1/2)")
print(f"  screens (peaked T_i opposes pinch): {bool(scr.screens)}")
print(f"  d Gamma_z / d(dT_i/dpsiHat): AD = {ad:.6e}, FD = {fd:.6e}, rel dev = {ad_fd_rel:.2e}")

# ----------------------------------------------------------------------------
# 5) JSON provenance
# ----------------------------------------------------------------------------
scan_sorted = [cache["scan"][f"{nu_n:.6e}"] for nu_n in NU_N_SCAN]
record = {
    "case": "classical_and_neoclassical_impurity_transport",
    "equilibrium": EQUILIBRIUM,
    "rN_wish": RN_WISH,
    "resolution": {"Ntheta": N_THETA, "Nzeta": N_ZETA, "Nxi": N_XI, "Nx": N_X,
                   "solverTolerance": SOLVER_TOLERANCE},  # fmt: skip
    "bulk": {"Z": BULK_Z, "mHat": BULK_M, "nHat": BULK_N, "THat": BULK_T},
    "impurity": {"Z": IMP_Z, "mHat": IMP_M, "nHat": IMP_N, "THat": IMP_T},
    "fortran_parity": fort,
    "collisionality_scan": scan_sorted,
    "classical_module_vs_dkx_max_rel_err": consistency,
    "charge_state_scan": {
        "Z": Z_SCAN,
        "nu_n": Z_SCAN_NU_N,
        "impurity_nHat": Z_SCAN_N_HAT,
        "classical_flux": [float(x) for x in pf_z],
        "classical_diffusion": [float(x) for x in np.asarray(dcl_z)],
        "screening_coefficient": screening_coeffs,
    },
    "screening": {
        "ion_density_peaking_coefficient": float(scr.ion_density_peaking_coefficient),
        "screening_coefficient": float(scr.screening_coefficient),
        "screens": bool(scr.screens),
        "d_gamma_d_dTi_AD": ad,
        "d_gamma_d_dTi_FD": fd,
        "ad_fd_relative_deviation": ad_fd_rel,
    },
    "references": [
        "S.I. Braginskii, Rev. Plasma Phys. 1, 205 (1965)",
        "P.H. Rutherford, Phys. Fluids 17, 1782 (1974)",
        "F.L. Hinton & R.D. Hazeltine, Rev. Mod. Phys. 48, 239 (1976)",
        "K.-D. Wenzel & D.J. Sigmar, Nucl. Fusion 30, 1117 (1990)",
        "P. Helander & D.J. Sigmar, Collisional Transport in Magnetized Plasmas, CUP (2002)",
    ],
}
JSON_PATH.write_text(json.dumps(record, indent=1))
print(f"\nStep 5: wrote {JSON_PATH}")

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
        "legend.fontsize": 9.0,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
colors = ["#1f77b4", "#d62728", "#2ca02c"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)

nu = np.asarray([r["nu_n"] for r in scan_sorted])
neo = np.asarray([r["impurity_neoclassical_flux"] for r in scan_sorted])
cls = np.asarray([r["impurity_classical_flux_module"] for r in scan_sorted])
total = neo + cls
linthresh = 0.3 * float(np.min(np.abs(cls)))
ax1.set_xscale("log")
ax1.set_yscale("symlog", linthresh=linthresh)
ax1.axhline(0.0, color="0.6", lw=0.8, zorder=0)
ax1.plot(nu, cls, "-s", ms=4, color=colors[1], label="classical (algebraic)")
ax1.plot(nu, neo, "-o", ms=4, color=colors[0], label="neoclassical (kinetic solve)")
ax1.plot(nu, total, "-^", ms=4, color=colors[2], label="total")
ax1.set_xlabel(r"$\nu_n$")
ax1.set_ylabel(r"$\Gamma_z \cdot (\nabla\psi)$  (v3 units; $>0$ outward)")
ax1.set_title("Impurity flux vs collisionality (C$^{6+}$, W7-X)")
ax1.legend(frameon=False, loc="upper left")

z = np.asarray(Z_SCAN)
ax2.loglog(z, np.abs(pf_z), "-o", ms=4, color=colors[2], label=r"$|\Gamma_z^{\rm cl}|$")
ax2.set_xlabel(r"impurity charge $Z$")
ax2.set_ylabel(r"$|\Gamma_z^{\rm cl}|$  (v3 units)")
ax2.set_title("Classical Z-scaling and screening")
axr = ax2.twinx()
axr.plot(z, screening_coeffs, "-^", ms=4, color=colors[1])
axr.axhline(0.5, color="0.6", lw=0.8, ls="--")
axr.set_ylabel(r"screening coeff $H$", color=colors[1])
axr.tick_params(axis="y", labelcolor=colors[1])
axr.set_ylim(0.0, 0.6)
axr.grid(False)
ax2.legend(frameon=False, loc="upper left")

fig.suptitle(
    "Classical / neoclassical impurity transport: mixed-collisionality benchmark", y=1.05
)
fig.savefig(PNG_PATH, dpi=170, bbox_inches="tight")
plt.close(fig)

img = Image.open(PNG_PATH).convert("P", palette=Image.ADAPTIVE, colors=64)
img.save(PNG_PATH, optimize=True)
print(f"  wrote {PNG_PATH} ({PNG_PATH.stat().st_size / 1024:.0f} KB)")
print("Done.")
