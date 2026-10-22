"""DKX monoenergetic coefficients on the W7-X standard configuration against the ICNTS data.

``validation/beidler2011_w7x_monoenergetic.csv`` holds D11*, D31* and D33* of Beidler et al.,
Nucl. Fusion 51, 076001 (2011), Figs 17, 26 and 21 (rho = 0.5), digitized from the paper. The
NEO-2 circles are the reference at v_E = 0; at v_E > 0 the reference is the cluster of DKES and
Monte Carlo symbols, whose half extent is the scatter between the codes.

Tolerances: D11* 10% (3% digitization, the rest the scatter of the ICNTS codes), D31* 0.01
absolute, D33* 2.5%. DKX is on the repository's ``w7x_standardConfig.bc`` (iota 0.8719 against
0.8700 in the paper), so the comparison is also a check on that equilibrium. Measured deviations
at the test grids (laptop, 4 threads): D11* +1.6 to +3.0% (nu* 1.1e-3 to 1.1), D31* <= 0.007,
D33* <= 1.5%; at v_E = 3e-3, nu* = 6.2e-5 and 1.8e-4: -5% and +10% of the cluster centre. At a
finer grid (29 x 65 x 128) D11* is +5.4% at nu* = 1.1e-3 and +7.7% at 3.6e-4.

DKX's D31* takes the sign of iota (this file stores iota < 0), so Beidler's sign convention,
D31* -> +1 in a tokamak banana limit, is -D31*.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
# nu*/nuPrime and v_E/EStar of this surface (iota 0.87185, R0 = 5.788, eps_t = 0.04531); asserted below.
NU_STAR_PER_NU_PRIME = 0.9589122762072304
V_E_PER_E_STAR = 0.87185 * 0.04530634904875981
TOL = {"D11": ("rel", 0.10), "D31": ("abs", 0.01), "D33": ("rel", 0.025)}
DECK = """&general
  RHSMode = 3
/
&geometryParameters
  geometryScheme = 11
  equilibriumFile = "w7x_standardConfig.bc"
  inputRadialCoordinate = 3
  rN_wish = 0.5
/
&speciesParameters
/
&physicsParameters
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {nt}
  Nzeta = {nz}
  Nxi = {nxi}
  Nx = 1
  solverTolerance = 1d-8
/
"""


def _reference() -> list[dict]:
    lines = (ROOT / "validation/beidler2011_w7x_monoenergetic.csv").read_text().splitlines()
    return list(csv.DictReader(line for line in lines if not line.startswith("#")))


def _database(tmp_path: Path, grid: tuple[int, int, int], nu_star: float, v_e: float):
    from dkx.monoenergetic import monoenergetic_database

    deck = tmp_path / "input.namelist"
    deck.write_text(DECK.format(nt=grid[0], nz=grid[1], nxi=grid[2]))
    db = monoenergetic_database(
        deck, [nu_star / NU_STAR_PER_NU_PRIME], [v_e / V_E_PER_E_STAR], tol=1e-8
    )
    np.testing.assert_allclose(np.asarray(db.nu_star), [nu_star], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(db.v_e), [v_e], rtol=1e-6, atol=1e-12)
    return {
        "D11": float(np.ravel(db.d11_star)[0]),
        "D31": -float(np.ravel(db.d31_star)[0]),
        "D33": float(np.ravel(db.d33_star)[0]),
    }


def _rows(v_e: float, nu_stars: list[float], coefficients=("D11", "D31", "D33")) -> dict:
    out: dict = {}
    for r in _reference():
        if float(r["v_E"]) == v_e and float(r["nu_star"]) in nu_stars and r["coef"] in coefficients:
            out[(float(r["nu_star"]), r["coef"])] = (float(r["value"]), float(r["abs_err"]))
    return out


@pytest.mark.parametrize("nu_star", [1.09e-3, 1.09e-2, 1.09e-1, 1.09])
def test_w7x_standard_matches_neo2_at_zero_electric_field(tmp_path: Path, nu_star: float) -> None:
    got = _database(tmp_path, (21, 45, 64), nu_star, 0.0)
    for (_, coef), (ref, _err) in _rows(0.0, [nu_star]).items():
        kind, tol = TOL[coef]
        dev = abs(got[coef] - ref) / (abs(ref) if kind == "rel" else 1.0)
        assert dev < tol, (nu_star, coef, got[coef], ref)


@pytest.mark.parametrize(("v_e", "nu_star"), [(3e-3, 6.2e-5), (3e-3, 1.82e-4)])
def test_w7x_standard_is_inside_the_code_scatter_at_finite_electric_field(
    tmp_path: Path, v_e: float, nu_star: float
) -> None:
    got = _database(tmp_path, (25, 55, 96), nu_star, v_e)
    ref, half_extent = _rows(v_e, [nu_star], ("D11",))[(nu_star, "D11")]
    assert abs(got["D11"] - ref) < half_extent, (got["D11"], ref, half_extent)
