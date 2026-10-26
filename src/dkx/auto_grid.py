"""Automatic phase-space resolution from the field's geometry and the plasma's collisionality.

``[resolution] auto = true`` fills the sizes a case leaves out. The grid follows from two things
read from the case:

* the **field**, from the Boozer-space spectrum of ``|B|`` on the case's surfaces: the dominant
  helicity ``(m, n)``, the rate ``w = |m iota + n N_fp|`` at which a field line crosses its wells,
  the effective major radius ``R = |B_zeta| / B_00`` and so the connection length
  ``L_c = R / w``, and the largest poloidal and toroidal mode numbers above 1e-2 of ``B_00``;
* the **plasma**, through the collisionality of every species,
  ``q_a = log10( Z_a^2 (sum_b Z_b^2 n_b) L_c / T_a^2 )`` with ``n`` in 1e20 m^-3, ``T`` in keV and
  ``L_c`` in metres (``lambda_a / L_c`` scales as ``T_a^2 / (Z_a^2 sum_b Z_b^2 n_b L_c)``). The
  lowest ``q`` over species and surfaces sets the grid, because the smallest mean free path is the
  hardest to resolve.

The pitch resolution follows ``N_xi = N_0 10^(-0.45 q)`` (the measured slope of the converged
references, whose collisional boundary layer in pitch narrows with ``lambda / L_c``) with
``N_0 = 86`` for a field with toroidal structure and ``N_0 = 457`` for an axisymmetric one,
whose trapped-passing boundary layer is the banana one. The result is rounded up to the next rung
of a ladder of joint grids, whose angular and speed sizes grow with it; the angular sizes are also
raised to ``4 m_max + 7`` and ``4 n_max + 9`` for a field with a broad spectrum.

The finest rung of each ladder is the *safe finer grid*, not a certified one: no reference
converged to 5% could be computed at that collisionality at a reasonable cost (the W7-X and
HSX references at 21 x 49 x 96 x 8 need 36 GB and more than an hour; the HSX electron heat flux moves by
14% between toroidal sizes of 33, 41 and 49 at fixed pitch). It is the answer whenever the case asks
for more pitch modes than the rung before it, and is flagged by :attr:`AutoGrid.beyond_ladder`; then
``dkx converge`` or ``--check-grid`` is the way to size the case. The constants are fitted to converged references on a tokamak, HSX, W7-X, a quasi-axisymmetric
and a quasi-helical field (``docs/user_guide/convergence.md``).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Joint grids (theta, zeta, pitch, speed) from coarse to fine; the last is the uncertified safe grid.
#: The first rung of the measured ladder (11, 13, 16, 4) is not used: a collisional case still needs
#: 13 x 17 x 24 x 5 for 3% on the measured fields.
STELLARATOR_RUNGS = ((13, 17, 24, 5), (15, 25, 32, 6), (17, 33, 48, 6), (19, 41, 64, 8), (21, 49, 96, 8))
TOKAMAK_RUNGS = ((13, 1, 24, 5), (25, 1, 48, 6), (49, 1, 96, 8), (65, 1, 160, 12), (81, 1, 256, 14))

#: Pitch modes needed at q = 0 and the slope of log10(N_xi) against q (fitted, see the module docstring).
PITCH_AT_Q0 = 86.0
PITCH_AT_Q0_AXISYMMETRIC = 457.0
PITCH_SLOPE = 0.45
MODE_THRESHOLD = 1.0e-2
AXISYMMETRIC_THRESHOLD = 1.0e-6

_TOKAMAK_ANALYTIC = "tokamak"
_ANALYTIC_PERIODS = {"tokamak": 10, "lhd_standard": 10, "lhd_inward": 10, "w7x_standard": 5}
_ANALYTIC_SCHEMES = {"tokamak": 1, "lhd_standard": 2, "lhd_inward": 3, "w7x_standard": 4}


@dataclass(frozen=True)
class FieldShape:
    """What the grid needs to know about ``|B|`` on one surface."""

    nfp: int
    iota: float
    major_radius_m: float
    eps: float  # amplitude of the dominant harmonic relative to B_00
    rate: float  # |m iota + n N_fp| of the dominant harmonic
    m_max: int  # largest poloidal mode number above MODE_THRESHOLD
    n_max: int  # largest toroidal mode number (per period) above MODE_THRESHOLD
    axisymmetric: bool

    @property
    def connection_length_m(self) -> float:
        return self.major_radius_m / max(self.rate, 1.0e-3)


@dataclass(frozen=True)
class AutoGrid:
    theta: int
    zeta: int
    pitch: int
    speed: int
    q: float  # lowest collisionality coordinate over species and surfaces
    pitch_wanted: float  # pitch modes the collisionality asks for
    beyond_ladder: bool  # the finest, uncertified rung was needed

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        return (self.theta, self.zeta, self.pitch, self.speed)


def _shape_from_surface(geometry: Any, nfp: int) -> FieldShape:
    b = np.asarray(geometry.b_hat, dtype=float)
    spectrum = np.fft.fft2(b) / b.size
    b00 = float(abs(spectrum[0, 0]))
    amp = np.abs(spectrum) / b00
    n_theta, n_zeta = b.shape
    iota = float(geometry.iota)
    best = (0.0, 0, 0)
    m_max = n_max = 0
    for m in range(0, n_theta // 2 + 1):
        for n in range(-(n_zeta // 2), n_zeta // 2 + 1):
            if m == 0 and n <= 0:
                continue
            a = float(amp[m % n_theta, n % n_zeta])
            if a > best[0]:
                best = (a, m, n)
            if a > MODE_THRESHOLD:
                m_max, n_max = max(m_max, m), max(n_max, abs(n))
    eps, m, n = best
    toroidal = float(np.max(amp[:, 1:])) if n_zeta > 1 else 0.0
    radius = abs(float(np.mean(np.asarray(geometry.b_hat_sub_zeta)))) / b00
    return FieldShape(
        nfp=int(nfp),
        iota=iota,
        major_radius_m=radius,
        eps=eps,
        rate=abs(m * iota + n * nfp) if eps > 0.0 else abs(iota),
        m_max=m_max,
        n_max=n_max,
        axisymmetric=bool(toroidal < AXISYMMETRIC_THRESHOLD),
    )


@functools.lru_cache(maxsize=8)
def _read(format_: str, path: str, stamp: float) -> Any:
    if format_ == "boozer":
        from dkx.magnetic_geometry import read_native_boozer  # noqa: PLC0415

        return read_native_boozer(path)
    from dkx.magnetic_geometry import read_vmec_wout  # noqa: PLC0415

    return read_vmec_wout(path)


def field_shapes(format_: str, file: Path, surfaces: Sequence[float], base_dir: Path) -> tuple[FieldShape, ...] | None:
    """Shape of ``|B|`` on each surface, or ``None`` when the geometry cannot be read."""
    from dkx.magnetic_geometry import FluxSurfaceGeometry  # noqa: PLC0415
    from dkx.phase_space import make_grids  # noqa: PLC0415

    try:
        name = str(file).lower().replace("-", "_")
        if format_ == "analytic":
            scheme, nfp, source = _ANALYTIC_SCHEMES[name], _ANALYTIC_PERIODS[name], None
        else:
            path = file if file.is_absolute() else base_dir / file
            source = _read(format_, str(path), path.stat().st_mtime)
            nfp = int(source.header.n_periods) if format_ == "boozer" else int(source.nfp)
        # The analytic tokamak is run on one toroidal point: its helical content is not part of the field.
        n_zeta = 1 if (format_ == "analytic" and name == _TOKAMAK_ANALYTIC) else 65
        grids = make_grids(
            n_theta=33, n_zeta=n_zeta, n_xi=4, n_x=2, n_l=2, n_periods=nfp,
            theta_derivative_scheme=0, zeta_derivative_scheme=0,
        )
        shapes = []
        for psi in surfaces:
            if source is None:
                geometry = FluxSurfaceGeometry.from_scheme(scheme, theta=grids.theta, zeta=grids.zeta)
            elif format_ == "boozer":
                geometry = FluxSurfaceGeometry.from_boozer_data(
                    source, theta=grids.theta, zeta=grids.zeta, r_n_wish=math.sqrt(psi), vmec_radial_option=0
                )
            else:
                geometry = FluxSurfaceGeometry.from_vmec(
                    source, theta=grids.theta, zeta=grids.zeta, psi_n_wish=psi, vmec_radial_option=0
                )
            shapes.append(_shape_from_surface(geometry, nfp))
        return tuple(shapes)
    except (OSError, KeyError, ValueError, AttributeError):
        return None


def collisionality_q(species: Sequence[Any], connection_length_m: float, surface: int) -> float:
    """Lowest ``q_a = log10(Z_a^2 sum_b Z_b^2 n_b L_c / T_a^2)`` over species at one surface."""
    density = np.array([s.density_m3[surface] for s in species], dtype=float) / 1.0e20
    temperature = np.array([s.temperature_keV[surface] for s in species], dtype=float)
    charge = np.array([s.charge for s in species], dtype=float)
    field = float(np.sum(charge**2 * density))
    q = charge**2 * field * connection_length_m / np.maximum(temperature, 1.0e-9) ** 2
    return float(np.log10(max(float(np.min(q)), 1.0e-30)))


def _odd(value: int) -> int:
    return value + 1 - value % 2


def choose_grid(species: Sequence[Any], shapes: Sequence[FieldShape] | None) -> AutoGrid:
    """The grid for the hardest surface of the case; the finest rung if the field is unknown."""
    if shapes is None:
        theta, zeta, pitch, speed = STELLARATOR_RUNGS[-1]
        return AutoGrid(theta, zeta, pitch, speed, q=float("nan"), pitch_wanted=float("nan"), beyond_ladder=True)
    q_all = [collisionality_q(species, shape.connection_length_m, i) for i, shape in enumerate(shapes)]
    worst = int(np.argmin(q_all))
    q, shape = q_all[worst], shapes[worst]
    axisymmetric = shape.axisymmetric
    wanted = (PITCH_AT_Q0_AXISYMMETRIC if axisymmetric else PITCH_AT_Q0) * 10.0 ** (-PITCH_SLOPE * q)
    rungs = TOKAMAK_RUNGS if axisymmetric else STELLARATOR_RUNGS
    fits = [rung for rung in rungs if rung[2] >= wanted]
    theta, zeta, pitch, speed = fits[0] if fits else rungs[-1]
    beyond = not fits or (theta, zeta, pitch, speed) == rungs[-1]
    theta = max(theta, _odd(4 * shape.m_max + 7))
    if not axisymmetric:
        zeta = max(zeta, _odd(4 * shape.n_max + 9))
    return AutoGrid(theta, zeta, pitch, speed, q=q, pitch_wanted=wanted, beyond_ladder=beyond)
