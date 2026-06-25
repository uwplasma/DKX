from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class CollisionalityLike(Protocol):
    """Record interface needed by validation collisionality math helpers."""

    label: str
    nuprime: float
    transport_matrix: np.ndarray


TRANSPORT_ELEMENTS: dict[str, tuple[int, int]] = {
    "L11": (0, 0),
    "L12": (0, 1),
    "L21": (1, 0),
    "L22": (1, 1),
    "L33": (2, 2),
}


def collisionality_grid(records: Sequence[CollisionalityLike]) -> list[float]:
    """Return the sorted normalized-collisionality grid in a scan."""

    return sorted({round(float(record.nuprime), 12) for record in records})


def collisionality_labels(records: Sequence[CollisionalityLike]) -> list[str]:
    """Return the sorted collision-model labels in a scan."""

    return sorted({record.label for record in records})


def l11_abs_series(records: Sequence[CollisionalityLike], *, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(nu', |L11|)`` for one collision model."""

    return transport_element_abs_series(records, label=label, element=TRANSPORT_ELEMENTS["L11"])


def transport_element_abs_series(
    records: Sequence[CollisionalityLike],
    *,
    label: str,
    element: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(nu', |L_ij|)`` for one collision model and matrix element."""

    selected = sorted((record for record in records if record.label == label), key=lambda record: record.nuprime)
    if not selected:
        raise ValueError(f"No collisionality records found for label {label!r}.")
    i, j = (int(element[0]), int(element[1]))
    nuprime = np.asarray([record.nuprime for record in selected], dtype=np.float64)
    values = np.asarray([abs(float(record.transport_matrix[i, j])) for record in selected], dtype=np.float64)
    return nuprime, values


def collisionality_power_law_slope(
    records: Sequence[CollisionalityLike],
    *,
    label: str,
    element: tuple[int, int],
    n_fit: int = 3,
) -> float:
    """Fit ``|L_ij| ~ (nu')**slope`` on the high-collisionality tail."""

    nuprime, values = transport_element_abs_series(records, label=label, element=element)
    n_fit = int(n_fit)
    if n_fit < 2:
        raise ValueError("n_fit must be at least 2.")
    if nuprime.size < n_fit:
        raise ValueError(f"Need at least {n_fit} records to fit a power-law slope.")
    tail_nu = nuprime[-n_fit:]
    tail_values = np.maximum(values[-n_fit:], np.finfo(float).tiny)
    return float(np.polyfit(np.log(tail_nu), np.log(tail_values), 1)[0])


def fp_pas_l11_separation(records: Sequence[CollisionalityLike]) -> list[dict[str, float]]:
    """Measure FP/PAS separation in ``L11`` across collisionality.

    The 2014 SFINCS paper uses these scans to show where pitch-angle scattering
    captures the dominant low-collisionality radial-transport physics and where
    momentum conservation matters at higher collisionality.
    """

    by_key = {(record.label, round(float(record.nuprime), 12)): record for record in records}
    rows: list[dict[str, float]] = []
    for nuprime in collisionality_grid(records):
        fp = by_key[("Fokker-Planck", nuprime)]
        pas = by_key[("PAS", nuprime)]
        fp_l11 = float(fp.transport_matrix[0, 0])
        pas_l11 = float(pas.transport_matrix[0, 0])
        abs_delta = abs(fp_l11 - pas_l11)
        rows.append(
            {
                "nuprime": float(nuprime),
                "fp_l11": fp_l11,
                "pas_l11": pas_l11,
                "abs_delta": float(abs_delta),
                "relative_to_fp": float(abs_delta / max(abs(fp_l11), np.finfo(float).tiny)),
            }
        )
    return rows


def high_collisionality_trend_summary(
    records: Sequence[CollisionalityLike],
    *,
    n_fit: int = 3,
) -> dict[str, object]:
    """Summarize high-collisionality power-law trends from a corrected scan artifact."""

    slopes: dict[str, dict[str, float]] = {}
    for label in collisionality_labels(records):
        slopes[label] = {
            name: collisionality_power_law_slope(records, label=label, element=element, n_fit=n_fit)
            for name, element in TRANSPORT_ELEMENTS.items()
        }
    pas_l11_l12_positive = all(slopes["PAS"][name] > 0.5 for name in ("L11", "L12"))
    fp_l11_l12_inverse_like = all(slopes["Fokker-Planck"][name] < -0.5 for name in ("L11", "L12"))
    return {
        "n_fit": int(n_fit),
        "nuprime_tail": collisionality_grid(records)[-int(n_fit) :],
        "slopes": slopes,
        "gates": {
            "pas_l11_l12_positive": bool(pas_l11_l12_positive),
            "fp_l11_l12_inverse_like": bool(fp_l11_l12_inverse_like),
        },
        "state": "asymptotic_trend_proxy" if fp_l11_l12_inverse_like else "needs_wider_high_nu_scan",
    }


def high_collisionality_slope_sensitivity(
    records: Sequence[CollisionalityLike],
    *,
    label: str = "Fokker-Planck",
    elements: Sequence[str] = ("L11", "L12"),
    n_fit_values: Sequence[int] = (2, 3, 4, 5),
) -> list[dict[str, object]]:
    """Return tail-slope fits for several fit-window lengths.

    This is used for the Simakov-Helander audit: a robust high-collisionality
    claim should not depend sensitively on whether the last two, three, or four
    scan points are used for the log-log fit.
    """

    rows: list[dict[str, object]] = []
    max_points = len([record for record in records if record.label == label])
    for n_fit in n_fit_values:
        if int(n_fit) < 2 or int(n_fit) > max_points:
            continue
        slopes = {
            element_name: collisionality_power_law_slope(
                records,
                label=label,
                element=TRANSPORT_ELEMENTS[element_name],
                n_fit=int(n_fit),
            )
            for element_name in elements
        }
        rows.append({"n_fit": int(n_fit), "slopes": slopes})
    return rows


def recommended_high_collisionality_nuprime_grid(
    current_grid: Sequence[float],
    *,
    min_nuprime_for_full_limit: float,
    points_per_decade: int = 4,
) -> list[float]:
    """Recommend additional ``nu'`` values for a full high-collisionality audit.

    The Simakov-Helander comparison is only defensible once the fitted tail is
    clearly in ``nu' >> 1``. This helper converts the current scan extent into a
    compact logarithmic extension that reaches at least one decade past the last
    checked point or the configured full-limit threshold, whichever is larger.
    """

    grid = np.asarray([float(v) for v in current_grid if np.isfinite(float(v)) and float(v) > 0.0], dtype=np.float64)
    if grid.size == 0:
        raise ValueError("current_grid must contain at least one positive finite nuprime value.")
    current_max = float(np.max(grid))
    required = float(min_nuprime_for_full_limit)
    if current_max >= required:
        return []
    target = max(required, 10.0 * current_max)
    n_points = max(2, int(np.ceil((np.log10(target) - np.log10(current_max)) * int(points_per_decade))) + 1)
    values = np.logspace(np.log10(current_max), np.log10(target), n_points)
    extension = [float(v) for v in values if v > current_max * (1.0 + 1.0e-12)]
    if not extension or extension[-1] < target * (1.0 - 1.0e-12):
        extension.append(float(target))
    return extension
