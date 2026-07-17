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
    from sfincs_jax.monoenergetic import monoenergetic_database

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
    from sfincs_jax.monoenergetic import monoenergetic_database

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
    from sfincs_jax.monoenergetic import monoenergetic_database

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

    from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist
    from sfincs_jax.inputs import load_sfincs_input
    from sfincs_jax.moments import rhsmode1_moments
    from sfincs_jax.solve import solve
    from sfincs_jax.writer import operator_containers

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
