"""Variational (entropy-production) bounds on the monoenergetic D11 coefficient.

Physics gates for :mod:`sfincs_jax.variational`: the upper/lower functionals
(Hirshman et al., Phys. Fluids 29, 2951 (1986); van Rij and Hirshman, Phys.
Fluids B 1, 563 (1989)) must bracket the computed ``transportMatrix[0][0]``
to solver-residual precision, the relative gap must shrink under grid
refinement, and the bounds must be tight at high collisionality where the
solution is collision dominated.

Measured reference values on the tiny scheme-1 monoenergetic deck (recorded
2026-07-12, float64, direct tier-1 solves):

======================  =========  ==========================================
configuration           gap        notes
======================  =========  ==========================================
(9, 9, 6),  nuPrime=1.2e-3  1.66e-2   base deck
(13, 13, 8), same nu        5.67e-3   one resolution notch: gap / 2.93
(17, 17, 10), same nu       1.38e-3   two notches: further / 4.12
(9, 9, 6),  nuPrime=10      2.03e-3   tight collisional regime
(17, 17, 10), nuPrime=10    2.41e-4   tight + refined
======================  =========  ==========================================

The midpoint identity ``(upper + lower)/2 = D11`` holds to ~1e-13 relative
(exact for an antisymmetric streaming operator), so the bracketing assertion
carries a roundoff slack only.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import pytest

from sfincs_jax.run import run_transport_matrix
from sfincs_jax.variational import monoenergetic_d11_bounds

REF_DECK = Path(__file__).parent / "ref" / "monoenergetic_PAS_tiny_scheme1.input.namelist"


def _write_deck(
    tmp_path: Path,
    *,
    name: str = "input.namelist",
    ntheta: int | None = None,
    nzeta: int | None = None,
    nxi: int | None = None,
    nu_prime: float | None = None,
) -> Path:
    text = REF_DECK.read_text()

    def sub(key: str, value: str, source: str) -> str:
        return re.sub(rf"(?mi)^\s*{key}\s*=.*$", f"  {key} = {value}", source)

    text = sub("saveMatricesAndVectorsInBinary", ".false.", text)
    if ntheta is not None:
        text = sub("Ntheta", str(ntheta), text)
    if nzeta is not None:
        text = sub("Nzeta", str(nzeta), text)
    if nxi is not None:
        text = sub("Nxi", str(nxi), text)
    if nu_prime is not None:
        text = sub("nuPrime", f"{nu_prime:.6e}".replace("e", "d"), text)
    path = tmp_path / name
    path.write_text(text)
    return path


def _slack(d11: float) -> float:
    # midpoint identity + moment/quadratic-form consistency are ~1e-13 relative
    return 1e-11 * abs(d11)


def test_bounds_bracket_d11_and_reach_writer(tmp_path: Path) -> None:
    out = tmp_path / "sfincsOutput.h5"
    run = run_transport_matrix(_write_deck(tmp_path), emit=None, out_path=out)
    d11 = float(run.transport_matrix[0, 0])

    assert run.d11_bounds is not None
    lower = float(run.d11_bounds["transportCoeffD11LowerBound"])
    upper = float(run.d11_bounds["transportCoeffD11UpperBound"])
    gap = float(run.d11_bounds["transportCoeffD11BoundGap"])

    assert lower - _slack(d11) <= d11 <= upper + _slack(d11)
    # measured gap 1.66e-2 on the (9, 9, 6) base deck
    assert 1e-4 < gap < 0.1
    assert gap == pytest.approx(abs(upper - lower) / abs(d11), rel=1e-12)

    # the JAX-only keys land in the output file
    with h5py.File(out, "r") as h5:
        assert float(h5["transportCoeffD11LowerBound"][()]) == pytest.approx(lower, rel=1e-14)
        assert float(h5["transportCoeffD11UpperBound"][()]) == pytest.approx(upper, rel=1e-14)
        assert float(h5["transportCoeffD11BoundGap"][()]) == pytest.approx(gap, rel=1e-14)

    # module-level recompute agrees with the run-surfaced values
    geometry = run.input.geometry
    bounds = monoenergetic_d11_bounds(
        run.operator,
        run.state_vectors[0],
        g_hat=geometry.g_hat,
        i_hat=geometry.i_hat,
        iota=geometry.iota,
        b0_over_bbar=geometry.b0_over_bbar,
    )
    assert float(bounds.lower) == pytest.approx(lower, rel=1e-12)
    assert float(bounds.upper) == pytest.approx(upper, rel=1e-12)
    # the entropy-production reconstruction of D11 matches diagnostics.F90
    assert float(bounds.d11) == pytest.approx(d11, rel=1e-10)


def test_bound_gap_shrinks_with_resolution(tmp_path: Path) -> None:
    run_lo = run_transport_matrix(_write_deck(tmp_path, name="lo.namelist"), emit=None)
    run_hi = run_transport_matrix(
        _write_deck(tmp_path, name="hi.namelist", ntheta=13, nzeta=13, nxi=8), emit=None
    )
    gap_lo = float(run_lo.d11_bounds["transportCoeffD11BoundGap"])
    gap_hi = float(run_hi.d11_bounds["transportCoeffD11BoundGap"])

    for run in (run_lo, run_hi):
        d11 = float(run.transport_matrix[0, 0])
        assert (
            float(run.d11_bounds["transportCoeffD11LowerBound"]) - _slack(d11)
            <= d11
            <= float(run.d11_bounds["transportCoeffD11UpperBound"]) + _slack(d11)
        )
    # measured: 1.66e-2 -> 5.67e-3 (factor 2.93) for one notch in (theta, zeta, xi)
    assert gap_hi < 0.5 * gap_lo


def test_bounds_tight_at_high_collisionality(tmp_path: Path) -> None:
    run = run_transport_matrix(_write_deck(tmp_path, nu_prime=10.0), emit=None)
    d11 = float(run.transport_matrix[0, 0])
    lower = float(run.d11_bounds["transportCoeffD11LowerBound"])
    upper = float(run.d11_bounds["transportCoeffD11UpperBound"])
    gap = float(run.d11_bounds["transportCoeffD11BoundGap"])

    assert lower - _slack(d11) <= d11 <= upper + _slack(d11)
    # measured 2.03e-3 at nuPrime=10 on the base (9, 9, 6) grid: the certificate
    # collapses when collisions dominate (f is near the collisional subspace).
    assert gap < 5e-3
