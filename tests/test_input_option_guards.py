"""Namelist settings that used to be dropped on the floor are now validated.

Three combinations changed the physics but reached no dkx check:

- ``force0RadialCurrentInEquilibrium``: dkx hardcodes the v3 default ``.true.``
  (``globalVariables.F90:65``) and writes it verbatim to the output, but a deck
  requesting ``.false.`` was ignored, so the ``BDotCurlB`` term of
  ``geometry.F90:291`` and the ``factor2`` drift-flux coefficient of
  ``diagnostics.F90:430-436`` silently went missing.
- ``xGrid_k`` combined with ``xGridScheme`` 2 or 6: those schemes pin a speed
  node at ``x=0``, where the weight ``exp(-x^2) x^k`` is zero for ``k>0``, so
  every plain-``dx`` quadrature weight divided by zero and the moments came out
  ``NaN``.  ``validateInput.F90:1094-1104`` forces ``xGrid_k=0`` there.
- ``xGrid_k`` combined with ``xDotDerivativeScheme = -2`` on the uniform (3/4)
  and Chebyshev (7/8) grids: the same singularity by a second route, since that
  scheme differentiates on a sub-grid that keeps the ``x=0`` node.  Upstream
  shares the bug and has no guard for it, so dkx refuses the combination.

The defaults (``force0RadialCurrentInEquilibrium=.true.``, ``xGrid_k=0``) were
already correct, so every test that pins the default path asserts bit-for-bit
equality rather than a tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dkx.drift_kinetic import kinetic_operator_build_from_namelist
from dkx.input_compat import (
    check_force0_radial_current_in_equilibrium,
    require_force0_radial_current_in_equilibrium,
)
from dkx.inputs import (
    SfincsInput,
    load_sfincs_input,
    parse_sfincs_input_text,
    sfincs_input_from_raw,
)
from dkx.namelist import read_sfincs_input
from dkx.phase_space import make_grids, make_speed_grid, speed_grid_diff_matrices
from dkx.xgrid import make_x_grid

REF = Path(__file__).parent / "ref"
# xGridScheme=2 (a node pinned at x=0), 1 species, geometryScheme=1: no
# equilibrium file to resolve and small enough to build in a fraction of a second.
XGRID2_DECK = REF / "pas_1species_PAS_Er_tiny_xgrid2.input.namelist"


def _deck_text(
    *,
    x_grid_scheme: int = 2,
    x_grid_k: float | None = None,
    x_dot_derivative_scheme: int | None = None,
) -> str:
    """The tiny deck with its speed-grid keys retargeted."""
    text = XGRID2_DECK.read_text()
    assert "xGridScheme = 2" in text  # the substitutions below must bite
    assert "xDotDerivativeScheme = 0" in text
    text = text.replace("xGridScheme = 2", f"xGridScheme = {x_grid_scheme}", 1)
    if x_dot_derivative_scheme is not None:
        text = text.replace(
            "xDotDerivativeScheme = 0",
            f"xDotDerivativeScheme = {x_dot_derivative_scheme}",
            1,
        )
    if x_grid_k is not None:
        text = text.replace(
            "&otherNumericalParameters", f"&otherNumericalParameters\n  xGrid_k = {x_grid_k!r}", 1
        )
    return text


def _write_deck(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "input.namelist"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# force0RadialCurrentInEquilibrium
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", ["geometryParameters", "physicsParameters", "general"])
def test_force0_radial_current_false_is_rejected_wherever_the_deck_puts_it(
    tmp_path: Path, group: str
) -> None:
    """It is a SFINCS global, not a namelist member, so any group must be caught."""
    text = XGRID2_DECK.read_text().replace(
        f"&{group}", f"&{group}\n  force0RadialCurrentInEquilibrium = .false.", 1
    )
    path = _write_deck(tmp_path, text)

    with pytest.raises(NotImplementedError, match="force0RadialCurrentInEquilibrium"):
        load_sfincs_input(path)
    with pytest.raises(NotImplementedError, match="BDotCurlB"):
        kinetic_operator_build_from_namelist(read_sfincs_input(path))


def test_force0_radial_current_rejection_names_the_missing_physics() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        require_force0_radial_current_in_equilibrium(False)

    message = str(excinfo.value)
    assert "geometry.F90" in message  # the BDotCurlB term
    assert "diagnostics.F90" in message  # the factor2 drift-flux coefficient
    assert "globalVariables.F90:65" in message  # where the .true. default lives


def test_force0_radial_current_false_is_rejected_by_from_params() -> None:
    """The flat constructor recognizes the name only to reject the unported value."""
    with pytest.raises(NotImplementedError, match="force0RadialCurrentInEquilibrium"):
        SfincsInput.from_params(geometryScheme=1, force0RadialCurrentInEquilibrium=False)

    # ... and does not mistake it for a typo of a typed field.
    inp = SfincsInput.from_params(geometryScheme=1, force0RadialCurrentInEquilibrium=True)
    assert inp.geometry.geometry_scheme == 1


def test_force0_radial_current_true_leaves_the_operator_bit_for_bit_unchanged(
    tmp_path: Path,
) -> None:
    """Spelling out the v3 default must not perturb a single coefficient."""
    baseline = kinetic_operator_build_from_namelist(read_sfincs_input(XGRID2_DECK))
    explicit_path = _write_deck(
        tmp_path,
        XGRID2_DECK.read_text().replace(
            "&geometryParameters",
            "&geometryParameters\n  force0RadialCurrentInEquilibrium = .true.",
            1,
        ),
    )
    explicit = kinetic_operator_build_from_namelist(read_sfincs_input(explicit_path))

    np.testing.assert_array_equal(
        np.asarray(explicit.operator.rhs()), np.asarray(baseline.operator.rhs())
    )
    np.testing.assert_array_equal(
        np.asarray(explicit.grids.x_weights), np.asarray(baseline.grids.x_weights)
    )


def test_decks_without_the_key_keep_the_true_default() -> None:
    assert check_force0_radial_current_in_equilibrium(read_sfincs_input(XGRID2_DECK)) is True
    assert require_force0_radial_current_in_equilibrium(None) is True
    assert load_sfincs_input(XGRID2_DECK).warnings == ()


# ---------------------------------------------------------------------------
# xGrid_k with a node pinned at x=0 (xGridScheme 2/6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x_grid_scheme", [2, 6])
@pytest.mark.parametrize("x_grid_k", [1.0, 2.0, 3.0])
def test_typed_validation_forces_xgrid_k_to_zero_for_schemes_2_and_6(
    x_grid_scheme: int, x_grid_k: float
) -> None:
    """validateInput.F90:1094-1104 overrides with a warning rather than stopping."""
    inp = sfincs_input_from_raw(
        parse_sfincs_input_text(
            _deck_text(x_grid_scheme=x_grid_scheme, x_grid_k=x_grid_k)
        )
    )

    assert inp.other.x_grid_k == 0.0
    assert any("xGrid_k" in warning for warning in inp.warnings)


@pytest.mark.parametrize("x_grid_scheme", [1, 5])
def test_the_other_polynomial_speed_grids_keep_the_requested_xgrid_k(
    x_grid_scheme: int,
) -> None:
    """No node at x=0 there, so ``xGrid_k`` is a legitimate knob and is left alone."""
    inp = sfincs_input_from_raw(
        parse_sfincs_input_text(_deck_text(x_grid_scheme=x_grid_scheme, x_grid_k=2.0))
    )

    assert inp.other.x_grid_k == 2.0
    assert inp.warnings == ()


@pytest.mark.parametrize("x_grid_scheme", [2, 6])
@pytest.mark.parametrize("x_grid_k", [1.0, 2.0, 3.0])
def test_make_grids_overrides_xgrid_k_instead_of_returning_nan_weights(
    x_grid_scheme: int, x_grid_k: float
) -> None:
    """The grid builder is the funnel every solve path shares, so it guards too."""
    kwargs = dict(n_theta=9, n_zeta=1, n_xi=4, n_x=5, n_l=2, n_periods=1)

    with pytest.warns(RuntimeWarning, match="xGrid_k"):
        grids = make_grids(x_grid_scheme=x_grid_scheme, x_grid_k=x_grid_k, **kwargs)
    expected = make_grids(x_grid_scheme=x_grid_scheme, x_grid_k=0.0, **kwargs)

    for name in ("x", "x_weights", "ddx", "d2dx2"):
        got = np.asarray(getattr(grids, name))
        assert np.all(np.isfinite(got)), f"{name} is not finite"
        # The override must land exactly on the k=0 grid, not merely near it.
        np.testing.assert_array_equal(got, np.asarray(getattr(expected, name)))


def test_the_operator_built_from_an_xgrid_k_deck_is_finite_and_matches_k_zero(
    tmp_path: Path,
) -> None:
    """End to end: the deck that used to produce NaN moments now solves as k=0."""
    baseline = kinetic_operator_build_from_namelist(read_sfincs_input(XGRID2_DECK))
    with pytest.warns(RuntimeWarning, match="xGrid_k"):
        overridden = kinetic_operator_build_from_namelist(
            read_sfincs_input(_write_deck(tmp_path, _deck_text(x_grid_k=1.0)))
        )

    rhs = np.asarray(overridden.operator.rhs())
    assert np.all(np.isfinite(rhs))
    np.testing.assert_array_equal(rhs, np.asarray(baseline.operator.rhs()))
    np.testing.assert_array_equal(
        np.asarray(overridden.grids.x_weights), np.asarray(baseline.grids.x_weights)
    )


def test_every_xgrid_k_consumer_in_one_build_agrees_on_the_override(tmp_path: Path) -> None:
    """The Fokker-Planck collision matrices read ``xGrid_k`` from the deck too.

    They must take the same override as the speed grid; a grid built at ``k=0``
    against collision matrices built at ``k=1`` would be quietly inconsistent
    rather than NaN, which is harder to notice than the original bug.
    """
    fokker_planck = XGRID2_DECK.read_text().replace(
        "collisionOperator = 1", "collisionOperator = 0", 1
    )
    baseline = kinetic_operator_build_from_namelist(
        read_sfincs_input(_write_deck(tmp_path / "k0", fokker_planck))
    )
    with pytest.warns(RuntimeWarning, match="xGrid_k"):
        overridden = kinetic_operator_build_from_namelist(
            read_sfincs_input(
                _write_deck(
                    tmp_path / "k1",
                    fokker_planck.replace(
                        "&otherNumericalParameters", "&otherNumericalParameters\n  xGrid_k = 1.0", 1
                    ),
                )
            )
        )

    probe = np.linspace(0.1, 1.0, baseline.operator.total_size)
    got = np.asarray(overridden.operator.apply(probe))
    assert np.all(np.isfinite(got))
    np.testing.assert_array_equal(got, np.asarray(baseline.operator.apply(probe)))


@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
def test_the_low_level_speed_grids_refuse_a_weight_that_vanishes_at_a_node(k: float) -> None:
    """Below ``make_grids`` the combination is an error, never silent ``inf``."""
    for grid in (
        make_speed_grid(n_x=5, k=k, include_point_at_x0=True),
        make_x_grid(n=5, k=k, include_point_at_x0=True),
    ):
        with pytest.raises(ValueError, match="vanishes at the node"):
            grid.dx_weights(k)

    # k=0 pins the supported path: the weight is 1 at x=0 and the call succeeds.
    weights = make_speed_grid(n_x=5, k=0.0, include_point_at_x0=True).dx_weights(0.0)
    assert np.all(np.isfinite(weights))


# ---------------------------------------------------------------------------
# xGrid_k with xDotDerivativeScheme = -2 (the second route to the same x=0 pole)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x_grid_scheme", [3, 4, 7, 8])
@pytest.mark.parametrize("x_grid_k", [1.0, 2.0])
def test_xdot_scheme_minus2_rejects_a_nonzero_xgrid_k_on_grids_with_a_node_at_x0(
    x_grid_scheme: int, x_grid_k: float
) -> None:
    """``xdot_diff_matrices`` differentiates on ``x[:-1]``, which keeps x=0.

    Upstream reaches the same pole and does not check for it, so this is a dkx
    refusal rather than a validateInput.F90 mirror.
    """
    kwargs = dict(n_theta=5, n_zeta=1, n_xi=4, n_x=5, n_l=2, n_periods=1, x_max=5.0)

    with pytest.raises(ValueError, match="xDotDerivativeScheme=-2 requires xGrid_k=0"):
        make_grids(
            x_grid_scheme=x_grid_scheme,
            x_grid_k=x_grid_k,
            x_dot_derivative_scheme=-2,
            **kwargs,
        )


@pytest.mark.parametrize("x_grid_scheme", [3, 4, 7, 8])
def test_xdot_scheme_minus2_at_the_default_xgrid_k_is_untouched(x_grid_scheme: int) -> None:
    """The default (``xGrid_k=0``) is well posed and must keep building."""
    grids = make_grids(
        n_theta=5, n_zeta=1, n_xi=4, n_x=5, n_l=2, n_periods=1, x_max=5.0,
        x_grid_scheme=x_grid_scheme, x_grid_k=0.0, x_dot_derivative_scheme=-2,
    )  # fmt: skip

    for name in ("ddx_xdot_plus", "ddx_xdot_minus"):
        assert np.all(np.isfinite(np.asarray(getattr(grids, name)))), name


@pytest.mark.parametrize("x_grid_scheme", [1, 5])
def test_xdot_scheme_minus2_still_accepts_xgrid_k_where_no_node_sits_at_x0(
    x_grid_scheme: int,
) -> None:
    """``xGrid_k`` is a real knob on the polynomial grids; only x=0 is fatal."""
    grids = make_grids(
        n_theta=5, n_zeta=1, n_xi=4, n_x=5, n_l=2, n_periods=1,
        x_grid_scheme=x_grid_scheme, x_grid_k=2.0, x_dot_derivative_scheme=-2,
    )  # fmt: skip

    assert np.all(np.asarray(grids.x) > 0.0)
    assert np.all(np.isfinite(np.asarray(grids.ddx_xdot_plus)))


def test_the_typed_loader_rejects_the_combination_before_any_grid_is_built() -> None:
    """A ported deck should fail at load time, naming the keys it must change."""
    with pytest.raises(ValueError, match="xDotDerivativeScheme=-2 requires xGrid_k=0"):
        sfincs_input_from_raw(
            parse_sfincs_input_text(
                _deck_text(x_grid_scheme=3, x_grid_k=1.0, x_dot_derivative_scheme=-2)
            )
        )


def test_schemes_2_and_6_take_the_override_rather_than_the_rejection() -> None:
    """The 2/6 override runs first, so those decks still load (as in Fortran)."""
    inp = sfincs_input_from_raw(
        parse_sfincs_input_text(
            _deck_text(x_grid_scheme=6, x_grid_k=1.0, x_dot_derivative_scheme=-2)
        )
    )

    assert inp.other.x_grid_k == 0.0
    assert any("xGrid_k" in warning for warning in inp.warnings)


def test_the_diff_matrices_refuse_a_vanishing_weight_at_the_node_they_are_given() -> None:
    """The floor under every route into the polynomial differentiation matrices."""
    with_x0 = np.array([0.0, 0.5, 1.5, 3.0])

    with pytest.raises(ValueError, match="vanishes at the node"):
        speed_grid_diff_matrices(with_x0, k=1.0)

    # k=0 keeps the weight at 1 where x=0, so the same nodes are fine.
    ddx, d2dx2 = speed_grid_diff_matrices(with_x0, k=0.0)
    assert np.all(np.isfinite(ddx)) and np.all(np.isfinite(d2dx2))
