"""Tutorial 20 -- comparing with YANCC, SFINCS and MONKES: every normalization in one table.

The codes solve the same monoenergetic equation and print different numbers.  This
script converts each reference into DKX's Beidler ``D*`` convention with the formulas
written out in the comments below (collisionality, E*, the radial dpsi/dr each code
uses, D*, SFINCS hat units), then prints coefficient, DKX, reference, converted
reference and relative error against a 2% gate.

Data: the YANCC (DSHAPE, NCSX) and MONKES (W7-X EIM) rows read the raw tables and the
sealed DKX values from ``validation/independent_cross_code_v1.json``, because those
equilibria are not shipped; the SFINCS row solves the committed tiny RHSMode 3 deck
live and compares it with the recorded Fortran v3 output.  Leaving the radial
convention out moves D11* by up to 7.5%, the size of the mismatch the table shows.

Run:              python examples/tutorials/20_cross_code_conversion.py
Runtime:          about 5 s on a laptop CPU, compilation included.
"""

# 1. Imports
import io
import json
import lzma
import math
from pathlib import Path

import h5py
import numpy as np

import dkx
from dkx.units import V_BAR

# 2. Input parameters
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "validation" / "independent_cross_code_v1.json"  # sealed YANCC/MONKES/DKX values
DECK = ROOT / "tests" / "ref" / "monoenergetic_PAS_tiny_scheme1.input.namelist"  # SFINCS deck
ORACLE = ROOT / "tests" / "ref" / "monoenergetic_PAS_tiny_scheme1.sfincsOutput.h5.xz"  # Fortran v3
OUT = ROOT / "examples" / "output" / Path(__file__).stem
GATE = 0.02  # relative-error gate per coefficient
E_STAR = 0.1  # DKX/SFINCS EStar converted to E_r and to the YANCC variables below
# YANCC's radial convention for DSHAPE: dpsi/dr = Phi_edge rho / (pi a_minor), from the
# equilibrium's edge toroidal flux (Wb) and minor radius (m) at the compared surface rho.
YANCC_DSHAPE = {"phi_edge": 1.0, "a_minor": 1.198365553577037}

# 3. Run
# Conventions used below (checked against dkx.monoenergetic, dkx.constants, the artifact's
# normalization_contract and the YANCC 0.0.3 source):
#  collisionality  YANCC and MONKES take nu/v in 1/m.  DKX nu' = |G + iota I| / B0 * (nu/v) / nuDHat(x0),
#                  nuDHat(x0) the shape factor of the applied Lorentz rate (0.83603 at x0 = 1);
#                  SFINCS nu_n = nuPrime B0 / (G + iota I).
#  electric field  YANCC E* = E_r / (v <B>), E_r = E_rho / a_minor in V/m (solver input E_rho / v in V s/m).
#                  DKX/SFINCS EStar = E_r / (v B0 iota eps_t), so E_r / (v B0) = EStar iota eps_t and
#                  dPhiHatdpsiHat = 2 EStar iota B0 / (alpha Delta G).  All comparisons here have E* = 0.
#  radial dpsi/dr  DKX B0 r_hat.  YANCC reports D_ij for rho and scales to DKES with a_minor, so its
#                  dpsi/dr = Phi_edge rho / (pi a_minor).  D11 scales with (psi_p / (B0 r_hat))^2,
#                  D31 and D13 with the ratio to the first power.
#  Beidler D*      D11* = D11 / D11_plateau, D31* = D31 / D31_banana, D33* = D33 / D33_Spitzer with
#                  R0 = |G| / B0, eps_t = r_hat / R0, 1 - f_c = 1.46 sqrt(eps_t),
#                  D33_Spitzer = 2 / (3 nu/v) <B^2> / B0^2.  D31* and D13* are positive in the banana
#                  limit for either sign of iota (orientation factor).
#  SFINCS          hat units: 1e20 m^-3, 1 keV, proton mass, 1 T, 1 m; Delta = m v / (e B R);
#                  psiHat = psi / (BBar RBar^2).  The RHSMode 3 transportMatrix is stored transposed
#                  (Fortran order); the single speed node (x0, w0) is deconvolved with
#                  (4 / sqrt(pi)) w0 x0^2 exp(-x0^2).
rows = []  # (case, coefficient, DKX, reference, converted reference)
radial_effect = {}  # D11* change from the radial convention, per case
for case in json.loads(ARTIFACT.read_text())["cases"]:
    norm, raw = case["normalization"], case["reference"]["raw_dkes"]
    g, iota, b0, r = norm["g_hat_T_m"], abs(norm["iota"]), norm["B0_T"], norm["local_r_m"]
    r_major = abs(g) / b0  # R0
    eps_t = r / r_major
    # Reference-code dpsi/dr over DKX's B0 r_hat; 1 when the table already uses B0 r_hat.
    psi_p = norm.get("psi_p_reference")
    if case["id"].startswith("dshape") and psi_p is None:
        psi_p = YANCC_DSHAPE["phi_edge"] * case["surface"]["rho"] / math.pi / YANCC_DSHAPE["a_minor"]
    ratio = 1.0 if psi_p is None else psi_p / (b0 * r)
    nu_v = case["equations"]["nu_over_v_per_m"]
    spitzer = raw["D33_spitzer"] if "D33_spitzer" in raw else 2 / (3 * nu_v) * raw["fsab_B2"] / norm["raw_B0_T"] ** 2
    cross = norm["cross_orientation"] * norm["raw_B0_T"] / b0 * 1.5 * iota * eps_t * b0 / (1.46 * math.sqrt(eps_t))
    converted = {
        "D11*": raw["D11"] * 8 * r_major * b0**2 * iota / math.pi * ratio**2,
        "D31*": raw["D31"] * cross * ratio,
        "D13*": raw["D13"] * cross * ratio,
        "D33*": raw["D33"] / spitzer,
    }
    radial_effect[case["id"].split("_")[0]] = abs(ratio**2 - 1)
    for key, value in converted.items():
        rows.append((case["reference"]["code"] + " " + case["id"].split("_")[0], key,
                     case["dkx"]["beidler"][key.replace("*", "_star")], raw[key[:-1]], value))

# SFINCS: solve the committed RHSMode 3 deck, read the recorded Fortran v3 output.
run = dkx.run_transport_matrix(DECK, emit=None)
with h5py.File(io.BytesIO(lzma.decompress(ORACLE.read_bytes()))) as f:
    sfincs = np.asarray(f["transportMatrix"][()]).T  # Fortran order -> row = radial, parallel
    c = {k: float(f[k][()]) for k in ("GHat", "IHat", "iota", "B0OverBBar", "Delta", "alpha", "FSABHat2", "psiHat", "nuPrime")}
x0, w0 = 1.0, float(np.asarray(run.operator.x_weights)[0])  # the single speed node
nu_d = 0.75 * math.sqrt(math.pi) * (math.erf(x0) - (math.erf(x0) - 2 * x0 / math.sqrt(math.pi) * math.exp(-x0**2)) / (2 * x0**2)) / x0**3
b0, r = c["B0OverBBar"], math.sqrt(2 * c["psiHat"] / c["B0OverBBar"])  # dpsiHat/drHat = B0 rHat
r_major, g_plus, sign = abs(c["GHat"]) / b0, abs(c["GHat"] + c["iota"] * c["IHat"]), math.copysign(1.0, c["iota"])
eps_t = r / r_major
v_d = c["Delta"] * x0**2 / (2 * r_major * b0)  # magnetic drift speed at the node
qfac = math.sqrt(math.pi) / (4 * w0 * x0**2 * math.exp(-(x0**2)))
d11_plateau = math.pi / 4 * v_d**2 * r_major / (x0 * abs(c["iota"]))
d31_banana = 2 / 3 * v_d * r_major / (abs(c["iota"]) * eps_t) * 1.46 * math.sqrt(eps_t)
nu_eff = c["nuPrime"] * b0 / g_plus * nu_d  # applied nu/v in hat units
d33_spitzer = x0**2 * c["FSABHat2"] / (3 * nu_eff * b0**2)
scales = {  # transportMatrix entry -> D*: (row, column, hat-unit factor, reference value)
    "D11*": (0, 0, -(c["Delta"] ** 2) / 4 / (b0 * r) ** 2 * c["GHat"] ** 2 / (b0 * g_plus), d11_plateau),
    "D13*": (0, 1, sign * c["Delta"] / 2 * abs(c["GHat"]) / b0 / (b0 * r), d31_banana),
    "D31*": (1, 0, -sign * c["Delta"] / 2 * abs(c["GHat"]) / b0 / (b0 * r), d31_banana),
    "D33*": (1, 1, g_plus / b0, d33_spitzer),
}
for key, (i, j, factor, reference) in scales.items():
    rows.append(("SFINCS tiny", key, qfac * factor * run.transport_matrix[i, j] / reference,
                 sfincs[i, j], qfac * factor * sfincs[i, j] / reference))

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
errors = [abs(dkx_value - converted) / abs(converted) for _, _, dkx_value, _, converted in rows]
lines = [f"{case:<14} {key:<5} {dkx_value:>13.6e} {ref:>13.6e} {conv:>13.6e} {err:>9.2e}  {'ok' if err <= GATE else 'FAIL'}"
         for (case, key, dkx_value, ref, conv), err in zip(rows, errors)]
(OUT / "table.txt").write_text("\n".join(lines) + "\n")
print(f"saved {OUT / 'table.txt'}")
print("\n=== Summary ===")
print(f"  {'case':<14} {'coef':<5} {'DKX':>13} {'reference':>13} {'converted':>13} {'rel. err':>9}")
for line in lines:
    print("  " + line)
print("  reference is the raw table value (YANCC/MONKES: DKES-scaled D_ij; SFINCS: transportMatrix entry);")
print("  converted is the same value in DKX's Beidler D* convention")
v_e = E_STAR * abs(c["iota"]) * eps_t  # E_r / (v B0)
print(f"\n  EStar={E_STAR:g} on the SFINCS deck: E_r/(v B0)={v_e:.4e} (YANCC E*), "
      f"E_r={v_e * V_BAR * b0 / 1e3:.4f} kV/m, "
      f"dPhiHatdpsiHat={2 * E_STAR * abs(c['iota']) * b0 / (c['alpha'] * c['Delta'] * c['GHat']):.4f}")
print(f"  largest relative error {max(errors):.2e} against the {GATE:.0%} gate: {'pass' if max(errors) <= GATE else 'FAIL'}")
print("  D11* shift if the radial convention (psi_p / (B0 r_hat))^2 were left out: "
      + ", ".join(f"{name} {shift:.1%}" for name, shift in radial_effect.items()))
