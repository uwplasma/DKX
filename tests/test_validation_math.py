from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from sfincs_jax.validation.artifacts import CollisionalityRecord
from sfincs_jax.validation.math import (
    TRANSPORT_ELEMENTS,
    collisionality_grid,
    collisionality_labels,
    collisionality_power_law_slope,
    fp_pas_l11_separation,
    high_collisionality_slope_sensitivity,
    high_collisionality_trend_summary,
    l11_abs_series,
    recommended_high_collisionality_nuprime_grid,
    transport_element_abs_series,
)


@dataclass(frozen=True)
class _RecordLike:
    label: str
    nuprime: float
    transport_matrix: np.ndarray


def _matrix(
    *,
    l11: float,
    l12: float,
    l21: float = 1.0,
    l22: float = 1.0,
    l33: float = 1.0,
) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[0, 0] = l11
    matrix[0, 1] = l12
    matrix[1, 0] = l21
    matrix[1, 1] = l22
    matrix[2, 2] = l33
    return matrix


def _power_law_records(nuprime_values: tuple[float, ...] = (1.0, 10.0, 100.0)) -> list[_RecordLike]:
    records: list[_RecordLike] = []
    for nuprime in nuprime_values:
        records.append(
            _RecordLike(
                label="PAS",
                nuprime=nuprime,
                transport_matrix=_matrix(
                    l11=2.0 * nuprime,
                    l12=3.0 * nuprime,
                    l21=4.0 * nuprime,
                    l22=5.0 * nuprime,
                    l33=6.0 * nuprime,
                ),
            )
        )
        records.append(
            _RecordLike(
                label="Fokker-Planck",
                nuprime=nuprime,
                transport_matrix=_matrix(
                    l11=-8.0 / nuprime,
                    l12=4.0 / nuprime,
                    l21=2.0 / nuprime,
                    l22=3.0 / nuprime,
                    l33=5.0 / nuprime,
                ),
            )
        )
    return records


def test_collisionality_grid_and_labels_are_sorted_and_rounded() -> None:
    records = [
        _RecordLike("PAS", 1.0000000000004, _matrix(l11=1.0, l12=1.0)),
        _RecordLike("Fokker-Planck", 10.0, _matrix(l11=1.0, l12=1.0)),
        _RecordLike("PAS", 1.0, _matrix(l11=2.0, l12=2.0)),
    ]

    assert collisionality_labels(records) == ["Fokker-Planck", "PAS"]
    assert collisionality_grid(records) == [1.0, 10.0]


def test_transport_series_sorts_by_nuprime_and_uses_absolute_values() -> None:
    records = [
        _RecordLike("PAS", 100.0, _matrix(l11=-9.0, l12=-2.0)),
        _RecordLike("PAS", 1.0, _matrix(l11=-3.0, l12=4.0)),
        _RecordLike("Fokker-Planck", 1.0, _matrix(l11=99.0, l12=99.0)),
    ]

    nuprime, values = transport_element_abs_series(records, label="PAS", element=TRANSPORT_ELEMENTS["L12"])
    l11_nuprime, l11_values = l11_abs_series(records, label="PAS")

    np.testing.assert_allclose(nuprime, [1.0, 100.0])
    np.testing.assert_allclose(values, [4.0, 2.0])
    np.testing.assert_allclose(l11_nuprime, [1.0, 100.0])
    np.testing.assert_allclose(l11_values, [3.0, 9.0])


def test_power_law_slope_validates_fit_window_and_label() -> None:
    records = _power_law_records()

    assert collisionality_power_law_slope(records, label="PAS", element=TRANSPORT_ELEMENTS["L11"]) == pytest.approx(
        1.0
    )
    assert collisionality_power_law_slope(
        records,
        label="Fokker-Planck",
        element=TRANSPORT_ELEMENTS["L12"],
    ) == pytest.approx(-1.0)

    with pytest.raises(ValueError, match="n_fit must be at least 2"):
        collisionality_power_law_slope(records, label="PAS", element=TRANSPORT_ELEMENTS["L11"], n_fit=1)
    with pytest.raises(ValueError, match="Need at least 4 records"):
        collisionality_power_law_slope(records, label="PAS", element=TRANSPORT_ELEMENTS["L11"], n_fit=4)
    with pytest.raises(ValueError, match="No collisionality records"):
        collisionality_power_law_slope(records, label="missing", element=TRANSPORT_ELEMENTS["L11"])


def test_fp_pas_l11_separation_uses_tiny_floor_for_zero_fp_reference() -> None:
    records = [
        _RecordLike("Fokker-Planck", 1.0, _matrix(l11=0.0, l12=1.0)),
        _RecordLike("PAS", 1.0, _matrix(l11=2.0, l12=1.0)),
    ]

    separation = fp_pas_l11_separation(records)

    assert separation == [
        {
            "nuprime": 1.0,
            "fp_l11": 0.0,
            "pas_l11": 2.0,
            "abs_delta": 2.0,
            "relative_to_fp": pytest.approx(2.0 / np.finfo(float).tiny),
        }
    ]


def test_high_collisionality_summary_and_sensitivity_gate_known_power_laws() -> None:
    records = _power_law_records(nuprime_values=(1.0, 10.0, 100.0, 1000.0))

    summary = high_collisionality_trend_summary(records, n_fit=3)
    sensitivity = high_collisionality_slope_sensitivity(records, n_fit_values=(1, 2, 5))

    assert summary["nuprime_tail"] == [10.0, 100.0, 1000.0]
    assert summary["gates"] == {
        "pas_l11_l12_positive": True,
        "fp_l11_l12_inverse_like": True,
    }
    assert summary["state"] == "asymptotic_trend_proxy"
    assert summary["slopes"]["PAS"]["L22"] == pytest.approx(1.0)
    assert summary["slopes"]["Fokker-Planck"]["L33"] == pytest.approx(-1.0)
    assert sensitivity == [
        {
            "n_fit": 2,
            "slopes": {
                "L11": pytest.approx(-1.0),
                "L12": pytest.approx(-1.0),
            },
        }
    ]


def test_recommended_high_collisionality_grid_filters_bad_values_and_reaches_target() -> None:
    extension = recommended_high_collisionality_nuprime_grid(
        [0.0, -1.0, float("nan"), 2.0],
        min_nuprime_for_full_limit=5.0,
        points_per_decade=2,
    )

    assert extension[0] > 2.0
    assert extension[-1] >= 20.0
    assert all(b > a for a, b in zip(extension, extension[1:]))
    assert (
        recommended_high_collisionality_nuprime_grid(
            [2.0, 50.0],
            min_nuprime_for_full_limit=5.0,
        )
        == []
    )
    with pytest.raises(ValueError, match="at least one positive finite"):
        recommended_high_collisionality_nuprime_grid([0.0, -1.0], min_nuprime_for_full_limit=5.0)


def test_validation_artifacts_collisionality_record_still_satisfies_extracted_math_api() -> None:
    records = [
        CollisionalityRecord("PAS", 1.0, _matrix(l11=2.0, l12=3.0)),
        CollisionalityRecord("PAS", 10.0, _matrix(l11=20.0, l12=30.0)),
    ]

    assert collisionality_power_law_slope(records, label="PAS", element=TRANSPORT_ELEMENTS["L11"], n_fit=2) == (
        pytest.approx(1.0)
    )
