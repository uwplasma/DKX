"""Shaing-Callen collisionless-limit physics tests for the D31 coefficient.

The RHSMode=3 bootstrap entry ``transportMatrix[1][0]`` must approach the
collisionless Shaing-Callen value (Shaing and Callen, Phys. Fluids 26, 3315
(1983); closed form as in Albert et al., arXiv:2407.21599, eq. (42)) as
``nuPrime -> 0``.  Convergence is slow (a ~sqrt(nu) collisional boundary
layer in the axisymmetric case) and non-monotonic in general geometry, so
the tests assert a monotone *approach* within envelopes chosen from the
measured tables below (recorded 2026-07-12, float64, tier-1 direct solves).

Axisymmetric scheme-1 deck (epsilon_t=0.15, iota=0.4542, GHat=3.7481,
IHat=0): the limit reduces to the Boozer-Gardner tokamak value
``lambda_bB = (GHat/iota) f_t`` and evaluates to D31_limit = -1.78601 in
transportMatrix units.  Measured D31/D31_limit:

    nuPrime   (Ntheta, Nzeta, Nxi)   D31            ratio     deficit
    3e-1      (17, 1, 16)            -2.3150e-2     0.0130
    1e-1      (21, 1, 20)            -1.5137e-1     0.0848
    3e-2      (25, 1, 28)            -5.5674e-1     0.3117
    1e-2      (29, 1, 40)            -1.0366e+0     0.5804    0.4196
    3e-3      (33, 1, 56)            -1.3714e+0     0.7679    0.2321
    1e-3      (37, 1, 80)            -1.5470e+0     0.8662    0.1338
    3e-4      (45, 1, 120)           -1.6572e+0     0.9279    0.0721
    1e-4      (55, 1, 160)           -1.7133e+0     0.9593    0.0407

The deficit (1 - ratio) scales like sqrt(nuPrime): successive deficit ratios
1.73 / 1.86 / 1.77 against sqrt(10/3) = 1.83 per half-decade-ish step.

Helical scheme-1 deck (the tiny reference geometry: epsilon_t=-0.07053,
epsilon_h=0.05067, (l, n)=(2, 10), iota=0.4542): the general evaluator gives
D31_limit = -0.61128.  Measured |D31 - limit| decreases monotonically:

    nuPrime   (Ntheta, Nzeta, Nxi)   D31            |D31 - limit|
    3e-2      (15, 15, 24)           -6.684e-2      0.545
    1e-2      (17, 17, 34)           -2.877e-1      0.324
    3e-3      (21, 21, 48)           -7.481e-1      0.137

The overshoot beyond the limit at low nu is the 1/nu-regime offset current
discussed by Albert et al.; equality is *not* asserted.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from dkx.run import run_transport_matrix
from dkx.shaing_callen import (
    shaing_callen_d31_limit,
    shaing_callen_lambda_bb,
    trapped_fraction,
)

HELICAL_DECK = Path(__file__).parent / "ref" / "monoenergetic_PAS_tiny_scheme1.input.namelist"

AXISYMMETRIC_DECK = """&general
  RHSMode = 3
/
&geometryParameters
  geometryScheme = 1
  epsilon_t = 0.15d+0
  epsilon_h = 0.0d+0
  iota = 0.4542d+0
  GHat = 3.7481d+0
  IHat = 0d+0
  helicity_l = 2
  helicity_n = 10
  B0OverBBar = 1d+0
/
&speciesParameters
/
&physicsParameters
  nuPrime = {nu_prime}
  EStar = 0.0d+0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {ntheta}
  Nzeta = 1
  Nxi = {nxi}
  NL = 3
  Nx = 1
  solverTolerance = 1d-12
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
"""


def _write_axisymmetric(tmp_path: Path, *, nu_prime: float, ntheta: int, nxi: int) -> Path:
    case = tmp_path / f"tok_nu{nu_prime:g}"
    case.mkdir()
    path = case / "input.namelist"
    path.write_text(
        AXISYMMETRIC_DECK.format(nu_prime=f"{nu_prime:.6e}".replace("e", "d"), ntheta=ntheta, nxi=nxi)
    )
    return path


def _write_helical(tmp_path: Path, *, nu_prime: float, ntheta: int, nzeta: int, nxi: int) -> Path:
    text = HELICAL_DECK.read_text()

    def sub(key: str, value: str, source: str) -> str:
        return re.sub(rf"(?mi)^\s*{key}\s*=.*$", f"  {key} = {value}", source)

    text = sub("saveMatricesAndVectorsInBinary", ".false.", text)
    text = sub("nuPrime", f"{nu_prime:.6e}".replace("e", "d"), text)
    text = sub("Ntheta", str(ntheta), text)
    text = sub("Nzeta", str(nzeta), text)
    text = sub("Nxi", str(nxi), text)
    case = tmp_path / f"hel_nu{nu_prime:g}"
    case.mkdir()
    path = case / "input.namelist"
    path.write_text(text)
    return path


def _d31_limit_from_operator(op, *, g_hat: float, i_hat: float, iota: float, n_periods: int) -> float:
    return float(
        shaing_callen_d31_limit(
            np.asarray(op.b_hat), g_hat=g_hat, i_hat=i_hat, iota=iota, n_periods=n_periods,
            x=np.asarray(op.x), x_weights=np.asarray(op.x_weights),
        ).d31  # fmt: skip
    )


def test_lambda_bb_axisymmetric_matches_trapped_fraction() -> None:
    """Axisymmetric reduction: lambda_bB = (GHat/iota) f_t (Boozer-Gardner).

    The pitch integral of eq. (42) collapses analytically to the trapped
    fraction for a B(theta)-only field; the general 2-D spectral evaluator
    must reproduce the 1-D quadrature to sub-percent accuracy.
    """
    g_hat, iota = 3.7481, 0.4542
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    b_hat = (1.0 + 0.15 * np.cos(theta))[:, None] * np.ones((1, 4))

    result = shaing_callen_lambda_bb(b_hat, g_hat=g_hat, i_hat=0.0, iota=iota, n_periods=1)
    expected = (g_hat / iota) * trapped_fraction(b_hat)
    assert result.lambda_bb == pytest.approx(expected, rel=1e-2)
    # both parts of the geometric factor are positive for this field
    assert result.term_passing > 0.0
    assert result.term_trapped > 0.0
    assert result.b_max == pytest.approx(1.15, rel=1e-12)


def test_d31_nu_scan_axisymmetric_smoke(tmp_path: Path) -> None:
    """Fast 2-point nu-scan: monotone approach to the Boozer-Gardner value.

    Envelopes from the measured table in the module docstring:
    ratio(nuPrime=1e-2) = 0.580, ratio(nuPrime=3e-3) = 0.768.
    """
    points = [(1e-2, 29, 40), (3e-3, 33, 56)]
    d31 = []
    op = None
    for nu_prime, ntheta, nxi in points:
        run = run_transport_matrix(
            str(_write_axisymmetric(tmp_path, nu_prime=nu_prime, ntheta=ntheta, nxi=nxi)), emit=None
        )
        d31.append(float(run.transport_matrix[1, 0]))
        op = run.operator

    limit = _d31_limit_from_operator(op, g_hat=3.7481, i_hat=0.0, iota=0.4542, n_periods=1)
    assert limit == pytest.approx(-1.78601, rel=1e-3)

    ratios = [value / limit for value in d31]
    assert all(np.sign(value) == np.sign(limit) for value in d31)
    # monotone approach from below along decreasing collisionality
    assert 0.0 < ratios[0] < ratios[1] < 1.0
    assert ratios[0] == pytest.approx(0.580, abs=0.03)
    assert ratios[1] == pytest.approx(0.768, abs=0.03)


def test_d31_nu_scan_axisymmetric_approaches_limit(tmp_path: Path) -> None:
    """Deep 3-point nu-scan: sqrt(nu) approach to the Shaing-Callen limit.

    Measured ratios 0.8662 / 0.9279 / 0.9593 at nuPrime = 1e-3 / 3e-4 / 1e-4;
    successive deficit ratios 1.86 and 1.77 (sqrt(nu) boundary layer).
    """
    points = [(1e-3, 37, 80), (3e-4, 45, 120), (1e-4, 55, 160)]
    d31 = []
    op = None
    for nu_prime, ntheta, nxi in points:
        run = run_transport_matrix(
            str(_write_axisymmetric(tmp_path, nu_prime=nu_prime, ntheta=ntheta, nxi=nxi)), emit=None
        )
        d31.append(float(run.transport_matrix[1, 0]))
        op = run.operator

    limit = _d31_limit_from_operator(op, g_hat=3.7481, i_hat=0.0, iota=0.4542, n_periods=1)
    ratios = np.asarray(d31) / limit
    deficits = 1.0 - ratios

    assert np.all(np.diff(ratios) > 0.0)
    assert np.all(deficits > 0.0)
    # final point within the measured envelope of the limit
    assert deficits[-1] < 0.06
    assert ratios[-1] == pytest.approx(0.959, abs=0.02)
    # sqrt(nu) scaling of the boundary-layer deficit across the scan steps
    shrink = deficits[:-1] / deficits[1:]
    assert np.all(shrink > 1.4)
    assert np.all(shrink < 2.4)


def test_d31_nu_scan_helical_qualitative(tmp_path: Path) -> None:
    """Helical scheme-1 geometry: monotone approach toward the general limit.

    In general geometry the 1/nu-regime offset does not vanish with nu
    (Albert et al., arXiv:2407.21599), so only the qualitative approach is
    asserted: |D31 - limit| decreases along the scan (measured 0.545 ->
    0.324 -> 0.137) and the last point lands within 30% of the limit
    (measured 22%).
    """
    points = [(3e-2, 15, 15, 24), (1e-2, 17, 17, 34), (3e-3, 21, 21, 48)]
    d31 = []
    op = None
    for nu_prime, ntheta, nzeta, nxi in points:
        run = run_transport_matrix(
            str(_write_helical(tmp_path, nu_prime=nu_prime, ntheta=ntheta, nzeta=nzeta, nxi=nxi)),
            emit=None,
        )
        d31.append(float(run.transport_matrix[1, 0]))
        op = run.operator

    limit = _d31_limit_from_operator(op, g_hat=3.7481, i_hat=0.0, iota=0.4542, n_periods=10)
    assert limit == pytest.approx(-0.61128, rel=2e-3)

    distances = np.abs(np.asarray(d31) - limit)
    assert np.all(np.diff(distances) < 0.0)
    assert distances[-1] < 0.30 * abs(limit)
    # the two lowest-collisionality points carry the sign of the limit
    assert np.sign(d31[-1]) == np.sign(limit)
    assert np.sign(d31[-2]) == np.sign(limit)
