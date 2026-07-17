"""CI-sized regression gate for the paper-benchmark monoenergetic scan.

Runs a tiny version of ``examples/paper_benchmarks/monoenergetic_icnts_w7x.py``
(2 nuPrime x 1 EStar on the W7-X standard configuration at reduced
resolution) and pins the structural physics the full benchmark relies on:

- shape/finiteness of the database arrays and positive ``D11*``/``nu_star``
  for a reversed-orientation Boozer equilibrium (``GHat < 0``, ``iota < 0``,
  ``psiHat < 0`` -- the orientation-robust normalization gate),
- the monotonic 1/nu rise of ``D11*`` at low collisionality,
- Onsager symmetry ``D13* = -D31*`` at ``EStar = 0``,
- one frozen-tolerance value so silent regressions in the solve, the
  geometry pipeline, or the normalization conversion surface here.

Measured on 2026-07-16 (float64, tier-1 direct solve, Ntheta=17, Nzeta=31,
Nxi=24, solverTolerance=1e-8): ``D11*(nuPrime=3e-3) = 0.20596213``,
``D11*(nuPrime=1e-2) = 0.16801237``, Onsager residual <= 2.7e-3.  Wall time
~15 s including the cached-equilibrium geometry load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
