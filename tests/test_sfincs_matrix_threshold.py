"""``SfincsMatrixThreshold`` reproduces SFINCS v3's matrix sparsification, and only on request.

SFINCS inserts every entry through ``sparsify.F90``, skipping ``|value| <= 1d-12``.
With hot electrons the ion->electron field-particle block falls below that level
(docs/experiments/2026-09-13-sfincs-sparsify-threshold.md), so the switch exists
for parity with such references and is off by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dkx.drift_kinetic import KineticOperator
from dkx.namelist import parse_sfincs_input_text

REF = Path(__file__).parent / "ref"
THRESHOLD = 1e-12


def _hot_electron_text() -> str:
    """The two-species FP fixture with an electron species far hotter than the ions."""
    text = (REF / "quick_2species_FPCollisions_noEr.input.namelist").read_text()
    replacements = {
        "Zs = 1 6": "Zs = 1 -1",
        "mHats = 1 6": "mHats = 1 5.4465d-4",
        "nHats = 0.6d+0 0.009d+0": "nHats = 0.6d+0 0.6d+0",
        "THats = 0.5d+0 0.8d+0": "THats = 0.06d+0 1.37d+0",
    }
    for old, new in replacements.items():
        assert old in text, old
        text = text.replace(old, new)
    return text


def _operator(text: str, extra: str | None = None) -> KineticOperator:
    if extra is not None:
        text = text + f"\n&otherNumericalParameters\n  {extra}\n/\n"
    return KineticOperator.from_namelist(parse_sfincs_input_text(text))


def test_the_default_keeps_every_collision_entry():
    text = _hot_electron_text()
    default = np.asarray(_operator(text).fp.mat)
    explicit_zero = np.asarray(_operator(text, "SfincsMatrixThreshold = 0d0").fp.mat)

    np.testing.assert_array_equal(default, explicit_zero)
    assert np.any((np.abs(default) <= THRESHOLD) & (default != 0.0))


def test_the_threshold_drops_only_small_entries_and_they_are_ion_to_electron():
    text = _hot_electron_text()
    full = np.asarray(_operator(text).fp.mat)
    thresholded = np.asarray(_operator(text, "SfincsMatrixThreshold = 1d-12").fp.mat)

    small = (np.abs(full) <= THRESHOLD) & (full != 0.0)
    np.testing.assert_array_equal(thresholded[small], 0.0)
    np.testing.assert_array_equal(thresholded[~small], full[~small])
    # (row species, column species): electrons are species 1, ions species 0.
    per_block = {(a, b): int(small[a, b].sum()) for a in range(2) for b in range(2)}
    assert per_block[(1, 0)] > 0
    assert per_block[(0, 0)] == per_block[(1, 1)] == 0


@pytest.mark.parametrize("value", ["-1d-12", "NaN"])
def test_a_negative_or_non_finite_threshold_is_refused(value):
    with pytest.raises(ValueError, match="SfincsMatrixThreshold"):
        _operator(_hot_electron_text(), f"SfincsMatrixThreshold = {value}")


def test_a_deck_without_a_fokker_planck_matrix_refuses_the_threshold():
    text = _hot_electron_text().replace("collisionOperator = 0", "collisionOperator = 1")
    with pytest.raises(ValueError, match="collisionOperator=0"):
        _operator(text, "SfincsMatrixThreshold = 1d-12")
