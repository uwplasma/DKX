"""Normalization and audit helpers for independent monoenergetic evidence.

DKX stores the Beidler et al. (2011) dimensionless coefficients, while
MONKES and YANCC expose the dimensional DKES-scaled geometric coefficients.
This module keeps the conversion used by the cross-code validation artifact
small, explicit, and independently testable.  It does not invoke either
external code and it is not part of the runtime solver path.
"""

from __future__ import annotations

import math
from typing import Mapping


_TRAPPED_FRACTION_FACTOR = 1.46


def nu_prime_for_nu_over_v(
    nu_over_v: float,
    *,
    g_hat: float,
    i_hat: float,
    iota: float,
    b0_over_bbar: float,
    nu_d_hat_x0: float,
) -> float:
    """Map the physical MDKE ``nu/v`` to DKX's ``nuPrime`` input.

    DKX applies ``nuPrime * B0 / |G + iota I| * nuDHat(x0)`` at the
    monoenergetic node.  YANCC and MONKES take the applied Lorentz frequency
    divided by speed directly, so matching equations requires the inverse
    map below rather than equating the two input numbers.
    """

    if b0_over_bbar == 0.0 or nu_d_hat_x0 <= 0.0:
        raise ValueError("b0_over_bbar and nu_d_hat_x0 must be nonzero and positive")
    g_plus = abs(float(g_hat) + float(iota) * float(i_hat))
    return g_plus / abs(float(b0_over_bbar)) * float(nu_over_v) / float(nu_d_hat_x0)


def dkes_to_beidler(
    *,
    d11: float,
    d31: float,
    d13: float,
    d33: float,
    nu_over_v: float,
    g_hat: float,
    iota: float,
    b0_over_bbar: float,
    r_hat: float,
    raw_b0_over_bbar: float | None = None,
    raw_fsab_b2: float | None = None,
    d33_spitzer: float | None = None,
    cross_orientation: float = -1.0,
) -> dict[str, float]:
    """Convert DKES-scaled coefficients to Beidler ``D*``.

    ``d11``/``d31``/``d13``/``d33`` are the values written by MONKES or
    returned by ``YANCC MDKESolution.get("Dij_DKES")``.  ``r_hat`` is the
    local effective radius, not the LCFS minor radius.  The optional
    ``raw_b0_over_bbar`` accounts for a reference-field choice in the raw
    cross coefficients that differs from DKX's recorded ``B0``.  Supply
    either the raw Spitzer value or ``raw_fsab_b2`` for the ``D33`` scale.

    The default ``cross_orientation=-1`` maps the handedness of the pinned
    YANCC/MONKES fixtures to DKX's orientation-standardized ``D31*`` and
    ``D13*``.  Diagonal coefficients are orientation invariant.
    """

    b0 = abs(float(b0_over_bbar))
    iota_abs = abs(float(iota))
    if b0 == 0.0 or iota_abs == 0.0 or r_hat <= 0.0:
        raise ValueError("B0, |iota|, and r_hat must be positive")
    r_major = abs(float(g_hat)) / b0
    if r_major == 0.0:
        raise ValueError("|g_hat| / B0 must be positive")
    eps_t = float(r_hat) / r_major
    if eps_t <= 0.0:
        raise ValueError("eps_t must be positive")

    d11_factor = 8.0 * r_major * b0 * b0 * iota_abs / math.pi
    d31_factor = (
        1.5
        * iota_abs
        * eps_t
        * b0
        / (_TRAPPED_FRACTION_FACTOR * math.sqrt(eps_t))
    )
    raw_b0 = b0 if raw_b0_over_bbar is None else abs(float(raw_b0_over_bbar))
    cross_b0_correction = raw_b0 / b0

    if (d33_spitzer is None) == (raw_fsab_b2 is None):
        raise ValueError("supply exactly one of d33_spitzer or raw_fsab_b2")
    if d33_spitzer is None:
        if nu_over_v <= 0.0 or raw_b0 == 0.0:
            raise ValueError("nu_over_v and raw B0 must be positive for the Spitzer scale")
        d33_spitzer = (
            2.0
            / (3.0 * float(nu_over_v))
            * float(raw_fsab_b2)
            / (raw_b0 * raw_b0)
        )
    if d33_spitzer == 0.0:
        raise ValueError("d33_spitzer must be nonzero")

    return {
        "D11_star": float(d11) * d11_factor,
        "D31_star": float(cross_orientation) * float(d31) * cross_b0_correction * d31_factor,
        "D13_star": float(cross_orientation) * float(d13) * cross_b0_correction * d31_factor,
        "D33_star": float(d33) / float(d33_spitzer),
        "r_major": r_major,
        "eps_t": eps_t,
        "D11_factor": d11_factor,
        "D31_factor": d31_factor,
        "cross_b0_correction": cross_b0_correction,
    }


def coefficient_relative_errors(
    candidate: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, float]:
    """Return absolute relative errors for the four Beidler coefficients."""

    errors: dict[str, float] = {}
    for key in ("D11_star", "D31_star", "D13_star", "D33_star"):
        denominator = abs(float(reference[key]))
        if denominator == 0.0:
            raise ValueError(f"reference {key} must be nonzero for a relative gate")
        errors[key] = abs(float(candidate[key]) - float(reference[key])) / denominator
    return errors


__all__ = ["coefficient_relative_errors", "dkes_to_beidler", "nu_prime_for_nu_over_v"]
