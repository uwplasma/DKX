"""Parity tests for the alternative speed grids and xDot derivative schemes.

xGridScheme 2 (Gauss-Radau), 3/4 (uniform grid on [0, xMax]), 7/8 (Chebyshev)
and the upwinded ``xDotDerivativeScheme`` variants (-2, 1-11) are canonical:
:func:`sfincs_jax.phase_space.make_grids` builds the nodes/weights/ddx pair
(ports of ``createGrids.F90``/``uniformDiffMatrices.F90``/``ChebyshevGrid.F90``)
and :class:`sfincs_jax.drift_kinetic.KineticOperator` applies the x=0 boundary
conditions (``pointAtX0``) and the upwinded E_r xDot term.  Each deck is pinned
against a Fortran SFINCS v3 golden, mirroring the sibling
``test_output_h5_scheme*_parity`` tests.  Grid first: the ``x`` nodes must match
at rounding level before the transport moments are compared.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import read_sfincs_h5

REF = Path(__file__).parent / "ref"

# (deck tag, xGridScheme, xDotDerivativeScheme)
CASES = [
    ("xgrid2", 2, 0),
    ("xgrid3", 3, 0),
    ("xgrid3_xdot1", 3, 1),
    ("xgrid3_xdot2", 3, 2),
    ("xgrid3_xdot3", 3, 3),
    ("xgrid3_xdot9", 3, 9),
    ("xgrid4", 4, 0),
    ("xgrid4_xdot4", 4, 4),
    ("xgrid4_xdot5", 4, 5),
    ("xgrid4_xdot6", 4, 6),
    ("xgrid4_xdot7", 4, 7),
    ("xgrid4_xdot8", 4, 8),
    ("xgrid4_xdot10", 4, 10),
    ("xgrid5_xdotm2", 5, -2),
    ("xgrid5_xdot11", 5, 11),
    ("xgrid7", 7, 0),
    ("xgrid8", 8, 0),
]

# Grid + scalar datasets compared bit-for-bit: the speed-grid convention gate.
_STRICT_KEYS = [
    "Nspecies", "Ntheta", "Nzeta", "Nxi", "NL", "Nx", "theta", "zeta", "x",
    "Nxi_for_x", "pointAtX0", "xGridScheme", "xGrid_k", "xMax", "Delta",
    "alpha", "nu_n", "Er", "dPhiHatdpsiHat", "collisionOperator",
    "constraintScheme", "RHSMode", "includeXDotTerm",
    "includeElectricFieldTermInXiDot", "BHat", "DHat",
]  # fmt: skip

# Kinetic transport outputs of the full canonical solve (state -> moments).
_TRANSPORT_KEYS = [
    "FSABFlow", "FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat",
    "momentumFlux_vm_psiHat", "particleFlux_vm0_psiHat", "heatFlux_vm0_psiHat",
    "NTV", "FSABFlow_vs_x",
]  # fmt: skip


def _paths(tag: str) -> tuple[Path, Path]:
    deck = REF / f"pas_1species_PAS_Er_tiny_{tag}.input.namelist"
    golden = REF / f"pas_1species_PAS_Er_tiny_{tag}.sfincsOutput.h5"
    assert golden.exists(), f"Missing Fortran fixture: {golden}"
    return deck, golden


@pytest.mark.parametrize(("tag", "x_grid_scheme", "x_dot_scheme"), CASES)
def test_xgrid_scheme_matches_fortran_fixture(
    tag: str, x_grid_scheme: int, x_dot_scheme: int, tmp_path: Path
) -> None:
    from sfincs_jax.run import run_profile

    deck, golden = _paths(tag)
    out = tmp_path / "sfincsOutput.h5"
    run = run_profile(deck, out_path=out, emit=None)
    assert run.solve_result.converged

    a = read_sfincs_h5(out)
    b = read_sfincs_h5(golden)
    assert a["input.namelist"] == b["input.namelist"]
    assert int(a["xGridScheme"]) == x_grid_scheme

    # Grid first: an x node/weight mismatch is a port bug, not a tolerance
    # problem, so the x grid and the scalar echoes gate at rounding level.
    strict = compare_sfincs_outputs(a_path=out, b_path=golden, keys=_STRICT_KEYS, rtol=0, atol=1e-12)
    bad = [r.key for r in strict if not r.ok]
    assert not bad, f"Mismatched strict keys for {tag}: {bad}"

    # Transport moments of the solved state (direct solves on both sides).
    transport = compare_sfincs_outputs(
        a_path=out, b_path=golden, keys=_TRANSPORT_KEYS, rtol=0, atol=1e-8
    )
    bad_t = [(r.key, r.max_abs) for r in transport if not r.ok]
    assert not bad_t, f"Mismatched transport keys for {tag}: {bad_t}"


@pytest.mark.parametrize(("tag", "x_grid_scheme", "x_dot_scheme"), CASES)
def test_xgrid_scheme_decks_route_through_canonical_stack(
    tag: str, x_grid_scheme: int, x_dot_scheme: int
) -> None:
    """No xGridScheme/xDotDerivativeScheme deck defers to the legacy pipeline."""
    from sfincs_jax.cli import deck_requires_legacy_pipeline
    from sfincs_jax.inputs import read_sfincs_input

    deck, _ = _paths(tag)
    assert deck_requires_legacy_pipeline(read_sfincs_input(deck)) is None
