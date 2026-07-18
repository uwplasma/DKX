"""CI-sized regression gates for the paper-benchmark monoenergetic scans.

Runs tiny versions of the ``examples/paper_benchmarks/monoenergetic_icnts_*``
cases (2 nuPrime x 1 EStar at reduced resolution) and pins the structural
physics the full benchmarks rely on:

- shape/finiteness of the database arrays and positive ``D11*``/``nu_star``
  across the stored orientations of the suite (W7-X: ``GHat < 0``,
  ``iota < 0``; TJ-II/HSX: ``GHat > 0``, ``iota < 0`` -- the
  orientation-robust normalization gate),
- the branch behavior of each machine (1/nu rise for W7-X and TJ-II, the
  plateau -> Pfirsch-Schlueter rise for near-quasisymmetric HSX),
- Onsager symmetry ``D13* = -D31*`` at ``EStar = 0``,
- one frozen-tolerance value per machine so silent regressions in the
  solve, the geometry pipeline, or the normalization conversion surface
  here.

The TJ-II deck is built from the Boozer ``|B|`` spectrum embedded in the
example script (parsed from its source; geometryScheme=13), so the tiny
gate also covers the namelist-spectrum path end to end.

Measured on 2026-07-16/17 (float64, tier-1 direct solves):
- W7-X (17x31x24): ``D11*(nuPrime=3e-3) = 0.20596213``, Onsager <= 2.7e-3,
  ~15 s including the cached-equilibrium geometry load.
- TJ-II (17x49x32): ``D11*(nuPrime=3e-2) = 13.74769803``, Onsager <= 0.085
  (the small TJ-II bootstrap coefficient converges slowly), ~8 s.
- HSX (11x61x32): ``D11*(nuPrime=1) = 1.73578230``, Onsager <= 1.9e-3 at
  nuPrime=1, ~15 s including the cached-equilibrium geometry load.
- Shaing-Callen mini-scan (W7-X 17x31x24): ``D31*(nuPrime=1e-2) =
  +9.82150972e-3``, ``D31*(nuPrime=3e-3) = -2.55865359e-2``, tiny-grid
  limit ``D31*_SC = -5.90226369e-2``; distances to the limit 6.884e-2 ->
  3.344e-2, ~11 s.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "paper_benchmarks"

DECK_TEMPLATE = """&general
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
  nuPrime = 1.0d+0
  EStar = 0.0d+0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 17
  Nzeta = 31
  Nxi = 24
  Nx = 1
  solverTolerance = 1d-8
/
&otherNumericalParameters
/
&preconditionerOptions
/
"""

NU_PRIMES = [3e-3, 1e-2]
E_STARS = [0.0]

# Frozen reference (see module docstring for the measurement record).
D11_STAR_FROZEN = 0.20596213


def test_tiny_w7x_monoenergetic_benchmark_scan(tmp_path: Path) -> None:
    from dkx.monoenergetic import monoenergetic_database

    deck = tmp_path / "monoenergetic_icnts_w7x_tiny.input.namelist"
    deck.write_text(DECK_TEMPLATE)
    db = monoenergetic_database(deck, NU_PRIMES, E_STARS)

    d11 = np.asarray(db.d11_star)
    d31 = np.asarray(db.d31_star)
    d13 = np.asarray(db.d13_star)
    d33 = np.asarray(db.d33_star)

    # Shape and finiteness of every coefficient table.
    for arr in (d11, d31, d13, d33):
        assert arr.shape == (len(NU_PRIMES), len(E_STARS))
        assert np.all(np.isfinite(arr))

    # Orientation-robust normalization: this .bc file stores GHat < 0,
    # iota < 0, psiHat < 0, yet the benchmark coefficients must be well
    # defined with positive references.
    assert float(db.g_hat) < 0.0
    assert float(db.iota) < 0.0
    assert db.r_major > 0.0
    assert db.eps_t > 0.0
    assert np.all(np.asarray(db.nu_star) > 0.0)
    assert np.all(d11 > 0.0)
    assert np.all(d33 > 0.0)

    # Monotonic 1/nu rise of D11* at low collisionality (EStar = 0).
    assert d11[0, 0] > d11[1, 0]

    # Onsager symmetry D13* = -D31* at EStar = 0 (couples the two
    # independently converted off-diagonal channels; measured <= 2.7e-3 at
    # this resolution).
    onsager = np.abs(d13 + d31) / np.abs(d31)
    assert np.max(onsager) < 1e-2

    # Frozen-tolerance regression value (deterministic direct solve; the
    # loose-ish tolerance absorbs BLAS/platform drift, not physics changes).
    assert abs(d11[0, 0] - D11_STAR_FROZEN) / D11_STAR_FROZEN < 1e-5


def _tjii_tiny_deck() -> str:
    """The TJ-II scheme-13 deck of the example script at tiny resolution.

    Parsed from the example source (not imported: the flat benchmark script
    executes on import), so the committed spectrum stays the single source
    of truth.
    """
    src = (EXAMPLES_DIR / "monoenergetic_icnts_tjii.py").read_text(encoding="utf-8")
    spectrum = re.search(r'TJII_BOOZER_BMNC = """\\\n(.*?)"""', src, re.S).group(1)
    template = re.search(r'DECK_TEMPLATE = """(.*?)"""', src, re.S).group(1)
    block = "\n".join("  " + line.strip() for line in spectrum.strip().splitlines())
    return template.format(
        rn_wish=0.70213959, spectrum=block, nu_prime=1.0, e_star=0.0,
        n_theta=17, n_zeta=49, n_xi=32, tol=1e-8,
    )  # fmt: skip


# Frozen references (module docstring for the measurement record).
TJII_D11_STAR_FROZEN = 13.74769803
HSX_D11_STAR_FROZEN = 1.73578230

HSX_DECK = """&general
  RHSMode = 3
/
&geometryParameters
  geometryScheme = 11
  equilibriumFile = "hsx3free.bc"
  inputRadialCoordinate = 3
  rN_wish = 0.5
/
&speciesParameters
/
&physicsParameters
  nuPrime = 1.0
  EStar = 0.0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 11
  Nzeta = 61
  Nxi = 32
  Nx = 1
  solverTolerance = 1d-8
/
&otherNumericalParameters
/
&preconditionerOptions
/
"""


def test_tiny_tjii_monoenergetic_benchmark_scan(tmp_path: Path) -> None:
    from dkx.monoenergetic import monoenergetic_database

    deck = tmp_path / "monoenergetic_icnts_tjii_tiny.input.namelist"
    deck.write_text(_tjii_tiny_deck())
    nu_primes = [3e-2, 1e-1]
    db = monoenergetic_database(deck, nu_primes, [0.0])

    d11 = np.asarray(db.d11_star)
    d31 = np.asarray(db.d31_star)
    d13 = np.asarray(db.d13_star)
    d33 = np.asarray(db.d33_star)
    for arr in (d11, d31, d13, d33):
        assert arr.shape == (len(nu_primes), 1)
        assert np.all(np.isfinite(arr))

    # Mixed stored orientation (GHat > 0, iota < 0) with positive references:
    # the complementary case to the W7-X (GHat < 0, iota < 0) gate above.
    assert float(db.g_hat) > 0.0
    assert float(db.iota) < 0.0
    assert db.r_major > 0.0 and db.eps_t > 0.0
    assert np.all(np.asarray(db.nu_star) > 0.0)
    assert np.all(d11 > 0.0)
    assert np.all(d33 > 0.0)

    # The strong-ripple 1/nu rise of D11* toward low collisionality.
    assert d11[0, 0] > d11[1, 0]

    # Onsager symmetry D13* = -D31* at EStar = 0.  The TJ-II bootstrap
    # coefficient converges slowly (measured residuals 0.085 and 0.021 at
    # this resolution), so the bound is loose but still catches sign or
    # normalization errors in either off-diagonal channel.
    onsager = np.abs(d13 + d31) / np.abs(d31)
    assert np.max(onsager) < 0.15

    # Frozen-tolerance regression value.
    assert abs(d11[0, 0] - TJII_D11_STAR_FROZEN) / TJII_D11_STAR_FROZEN < 1e-5


def test_tiny_hsx_monoenergetic_benchmark_scan(tmp_path: Path) -> None:
    from dkx.monoenergetic import monoenergetic_database

    deck = tmp_path / "monoenergetic_icnts_hsx_tiny.input.namelist"
    deck.write_text(HSX_DECK)
    nu_primes = [1.0, 10.0]
    db = monoenergetic_database(deck, nu_primes, [0.0])

    d11 = np.asarray(db.d11_star)
    d31 = np.asarray(db.d31_star)
    d13 = np.asarray(db.d13_star)
    d33 = np.asarray(db.d33_star)
    for arr in (d11, d31, d13, d33):
        assert arr.shape == (len(nu_primes), 1)
        assert np.all(np.isfinite(arr))

    # hsx3free.bc stores GHat > 0, iota < 0 in the v3 sign convention.
    assert float(db.g_hat) > 0.0
    assert float(db.iota) < 0.0
    assert db.r_major > 0.0 and db.eps_t > 0.0
    assert np.all(np.asarray(db.nu_star) > 0.0)
    assert np.all(d11 > 0.0)
    assert np.all(d33 > 0.0)

    # Near-quasisymmetry: no 1/nu rise here -- D11* increases from plateau
    # into the Pfirsch-Schlueter branch, and D33* approaches its collisional
    # limit of 1 from below as collisionality grows.
    assert d11[1, 0] > d11[0, 0]
    assert d33[1, 0] > d33[0, 0]
    assert abs(d33[1, 0] - 1.0) < 5e-3

    # Onsager symmetry at the plateau point (D31* at nuPrime=10 is ~7e-5 in
    # banana units, too small for a meaningful relative residual).
    onsager = np.abs(d13[0, 0] + d31[0, 0]) / np.abs(d31[0, 0])
    assert onsager < 1e-2

    # Frozen-tolerance regression value.
    assert abs(d11[0, 0] - HSX_D11_STAR_FROZEN) / HSX_D11_STAR_FROZEN < 1e-5


# Frozen reference for the Shaing-Callen mini-scan (module docstring).
SC_D31_STAR_FROZEN = -2.55865359e-2


def test_tiny_shaing_callen_convergence_scan(tmp_path: Path) -> None:
    """CI-sized version of examples/paper_benchmarks/shaing_callen_convergence.py.

    Two-point W7-X nuPrime scan at 17x31x24 pinning the structure the full
    benchmark relies on: D31* changes sign across the scan and moves
    *toward* the Shaing-Callen collisionless value (evaluated for the same
    surface, converted with the same normalization helper) as nuPrime
    decreases, plus one frozen-tolerance value.  ~11 s.
    """
    import numpy as np

    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.inputs import load_sfincs_input
    from dkx.monoenergetic import (
        monoenergetic_database,
        monoenergetic_dstar_from_transport_matrix,
    )
    from dkx.shaing_callen import shaing_callen_d31_limit

    deck = tmp_path / "shaing_callen_convergence_tiny.input.namelist"
    deck.write_text(DECK_TEMPLATE)
    nu_primes = [1e-2, 3e-3]  # descending: toward the collisionless limit
    db = monoenergetic_database(deck, nu_primes, [0.0])
    d31 = np.asarray(db.d31_star)[:, 0]
    assert np.all(np.isfinite(d31))

    # The Shaing-Callen limit for the same surface, in the same D31* units.
    op = kinetic_operator_from_namelist(load_sfincs_input(deck).raw)
    lim = shaing_callen_d31_limit(
        np.asarray(op.b_hat), g_hat=db.g_hat, i_hat=db.i_hat, iota=db.iota,
        n_periods=5, x=np.asarray(op.x), x_weights=np.asarray(op.x_weights),
    )  # fmt: skip
    tm = np.zeros((2, 2))
    tm[1, 0] = lim.d31
    point = monoenergetic_dstar_from_transport_matrix(
        tm, nu_prime=1.0, delta=db.delta, g_hat=db.g_hat, i_hat=db.i_hat,
        iota=db.iota, b0_over_bbar=db.b0_over_bbar,
        fsab_hat2=float(np.asarray(db.fsab_hat2)), r_hat=db.r_hat,
        x0=db.x0, w0=db.w0, nu_d_hat_x0=db.nu_d_hat_x0,
    )  # fmt: skip
    d31_limit = float(np.asarray(point.d31_star))
    assert abs(d31_limit - (-5.90226369e-2)) / abs(d31_limit) < 1e-3

    # Sign structure: positive on the plateau side, the limit's sign at the
    # lowest collisionality (the benchmark's sign change).
    assert d31_limit < 0.0
    assert d31[0] > 0.0
    assert np.sign(d31[1]) == np.sign(d31_limit)

    # |D31* - limit| decreases as nuPrime decreases: the approach toward
    # the collisionless asymptote (measured 6.884e-2 -> 3.344e-2).
    distances = np.abs(d31 - d31_limit)
    assert distances[1] < distances[0]

    # Frozen-tolerance regression value at the low-collisionality point.
    assert abs(d31[1] - SC_D31_STAR_FROZEN) / abs(SC_D31_STAR_FROZEN) < 1e-5


def test_tiny_gradient_verification_row(tmp_path: Path) -> None:
    """One cheap AD-vs-FD row from examples/paper_benchmarks/gradient_verification.py.

    Reproduces row (b) -- d(FSABjHat)/d(nu_n) through a differentiable
    RHSMode=1 solve -- at loose tolerance (measured rel dev 4.2e-9 on
    2026-07-17; asserted 1e-4).  Wall time ~5 s.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from dataclasses import replace

    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.inputs import load_sfincs_input
    from dkx.moments import rhsmode1_moments
    from dkx.solve import solve
    from dkx.writer import operator_containers

    src = (EXAMPLES_DIR / "gradient_verification.py").read_text(encoding="utf-8")
    pas_deck = re.search(r'PAS_DECK = """(.*?)"""', src, re.S).group(1)
    deck = tmp_path / "gradient_verification_tiny.input.namelist"
    deck.write_text(pas_deck)

    inp = load_sfincs_input(deck)
    op0 = kinetic_operator_from_namelist(inp.raw)
    nu0 = jnp.asarray(op0.pas.nu_n, dtype=jnp.float64)

    def fsabjhat(nu_n):
        pas = replace(op0.pas, nu_n=nu_n, coef=op0.pas.coef * (nu_n / op0.pas.nu_n))
        op = replace(op0, pas=pas)
        x = solve(op, op.rhs(), tol=1e-12, differentiable=True).x
        layout, vgrid, surface, species = operator_containers(op)
        table = rhsmode1_moments(
            layout, vgrid, surface, species, jnp.reshape(x, (-1,)),
            delta=op.delta, alpha=op.alpha,
        )  # fmt: skip
        return jnp.reshape(table["FSABjHat"], ())

    grad_ad = float(jax.grad(fsabjhat)(nu0))
    h = 1e-5 * float(nu0)
    grad_fd = float((fsabjhat(nu0 + h) - fsabjhat(nu0 - h)) / (2.0 * h))
    assert np.isfinite(grad_ad) and abs(grad_fd) > 0.0
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-4)
