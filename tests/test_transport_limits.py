"""Thermal transport limits proved against closed forms.

Two section-7.4 gaps are closed here, plus the high-collisionality limit of
the parallel-conductivity channel.  Nothing in this file re-proves work that
already exists: the *monoenergetic* Lorentz conductivity (``D33* -> 1``) is
``test_monoenergetic_database.py::test_normalization_physics_gates``, the
e-e momentum-restoring enhancement over the Lorentz value is
``test_collisions_improved_sugama.py::test_spitzer_momentum_restoring_factor``,
and the collisionless bootstrap limit is ``test_shaing_callen.py``.  What was
missing is the *Maxwellian-convolved* (thermal) statement and the convolution
machinery itself checked against manufactured analytic coefficients.

Derivations
===========

Normalization.  DKX uses ``vBar = sqrt(2 TBar/mBar)``, so in hat units the
thermal speed is ``vtHat = sqrt(THat/mHat)`` with *no* factor of two and the
speed variable is ``x = v/sqrt(2T/m)``; the Maxwellian is ``exp(-x^2)`` and
``fM = n mHat^{3/2}/(pi^{3/2} THat^{3/2}) exp(-x^2)`` (the normalization that
makes ``moments.rhsmode1_moments`` return ``densityPerturbation = n``).

1. Inductive L=1 response.  ``KineticOperator.rhs_phi1`` drives the L=1
   Legendre mode with

       rhs_1 = alpha Z x e^{-x^2} EParallelHat n mHat BHat
               / (pi^{3/2} THat^2 <BHat^2>),

   which is exactly the P_1 coefficient of ``(q E_par v xi / T) fM`` with the
   flux-function inductive field ``E_par = <E.B> BHat/<BHat^2>``.  With pitch
   angle scattering the L=1 eigenvalue of the collision operator is
   ``nu_n nuDHat(x)`` (``collisions.make_pitch_angle_scattering_v3_operator``
   uses ``L(L+1)/2``), and in a geometry with ``|B|`` constant the streaming
   and mirror terms annihilate the theta/zeta-independent response, so

       f_1 = rhs_1 / (nu_n nuDHat(x))   exactly.

   Feeding that through ``flow = (4 pi THat^2/(3 mHat^2)) sum_x w x^3 f_1``
   and ``FSABFlow = <BHat flow>/<...>`` (the ``BHat^2/<BHat^2>`` ratio is one
   for constant ``|B|``) gives the closed form

       FSABFlow = 4 alpha Z n EParallelHat / (3 sqrt(pi) mHat nu_n)
                  * sum_x w_x x^4 e^{-x^2} / nuDHat(x).                  (1)

   Equation (1) is *geometry free*: it contains no GHat, IHat, iota, B0 or
   Er.  It is the DKX-normalized parallel electrical conductivity.

2. Lorentz gas.  When the deflection frequency is a pure ``x^-3`` power law
   -- ``nuDHat = nuHat0/x^3``, the limit of
   ``collisions.nu_d_hat_pitch_angle_scattering_v3`` when every field species
   is infinitely heavy so ``erf(x_b) - Psi(x_b) -> 1`` -- the speed sum in (1)
   is a Gamma function.  The SFINCS speed grid is Gaussian for the weight
   ``x^k e^{-x^2}``, so with ``k = 0`` and ``Nx >= 4`` it integrates
   ``x^7 e^{-x^2}`` exactly:

       sum_x w x^7 e^{-x^2} = (1/2) Gamma(4) = 3,

   hence

       FSABFlow = 4 alpha Z n EParallelHat / (sqrt(pi) mHat nu_n nuHat0).  (2)

   The same sum with ``nuDHat`` frozen at the thermal speed (``x = 1``) is
   ``sum_x w x^4 e^{-x^2}/nuHat0 = 3 sqrt(pi)/(8 nuHat0)``, so the thermal
   convolution enhances the conductivity over its monoenergetic value at
   ``v = vth`` by exactly

       8/sqrt(pi) = 4.513516668382050.                                    (3)

   This is the classical Lorentz-conductivity factor.  Converting to the
   Braginskii collision time (``nu_D(vth) tau_e = (3/4) sqrt(pi)`` for the
   standard definitions) turns (3) into the textbook
   ``sigma = (32/(3 pi)) n e^2 tau_e / m``; the Spitzer value for Z = 1 is
   ``0.582`` times that, which is the separate e-e restoration factor already
   covered in ``test_collisions_improved_sugama.py``.

3. High collisionality (Pfirsch-Schlueter side).  Write the operator as
   ``V + C`` with ``V`` the streaming/mirror part (it maps Legendre L to
   L +- 1) and ``C`` the pitch-angle operator (diagonal in L).  Expanding the
   L=1 drive in ``V C^{-1}``, the first correction ``-C^{-1} V C^{-1} S``
   lives on L = 0 and 2 and does not contribute to the flow moment, so the
   leading correction to ``FSABFlow`` is ``(C^{-1}V)^2``: second order in
   ``1/nu``.  Each ``V`` acting on the L=1 response ``prop BHat`` produces one
   factor of ``grad_par BHat``, so the correction is also second order in the
   field-modulation amplitude ``epsilon_t``.  Its sign is fixed by the
   variational identity: for ``A = C + V`` with ``C`` symmetric positive
   definite and ``V`` antisymmetric,

       <S, C^{-1} S> = <S, A^{-1} S> + <Vy, C^{-1} Vy>,   y = A^{-T} S,

   so ``<S, A^{-1}S> <= <S, C^{-1}S>``: streaming (trapping) can only *reduce*
   the conductivity below the geometry-free value (1).

4. Monoenergetic-to-thermal convolution.  ``monoenergetic.energy_convolution``
   evaluates

       L_ij = (4/sqrt(pi)) int_0^inf dx x^2 e^{-x^2} D_ij(x) h_i h_j,
       h_1 = h_3 = 1, h_2 = x^2,

   with the kernels un-normalized from the stored ``D*`` tables by the Beidler
   reference values ``D11^p prop x^3``, ``D31^b prop x^2`` and
   ``D33^PS prop x^2/nuDHat(x)``.  Defining the Maxwellian moment

       M(m) = (4/sqrt(pi)) int_0^inf x^m e^{-x^2} dx
            = (2/sqrt(pi)) Gamma((m+1)/2),      M(2) = 1,

   any manufactured kernel that is a power of ``x`` integrates in closed form,
   which is what the two convolution tests below assert entry by entry.

Measured agreement (2026-08-31, float64, this file)
===================================================

=========================================================  ================
quantity                                                   measured
=========================================================  ================
FSABFlow vs closed form (1), constant |B|, 4 decks          <= 2.3e-16 rel
FSABFlow vs Lorentz Gamma form (2)                          1.6e-13 rel
thermal/monoenergetic conductivity ratio vs 8/sqrt(pi)      1.7e-13 rel
nu_n^-2 order of the trapping correction (nu_n 10->30->100) 1.9897, 1.9989
epsilon_t^2 order of the same (0.05->0.1->0.2)              1.9999, 1.9995
9 convolved L_ij vs Gamma closed form, power-law D*         1.5e-15 rel
8 convolved L_ij vs Gamma closed form, constant D*          8.0e-16 rel
convolved Lorentz L_33 and its 8/sqrt(pi) ratio             2.2e-15 rel
=========================================================  ================

The residual 1.6e-13 in (2) is the deliberate finite mass ratio of the
Lorentz gas (``mHat = 1e-12`` against a unit-mass background gives
``Psi(x sqrt(m_b/m_a)) ~ 1e-12`` contamination of ``erf - Psi``), not solver
error: the same run matches (1) -- which uses the *exact* ``nuDHat`` -- to
1.9e-16.
"""

from __future__ import annotations

from math import gamma, pi, sqrt
from pathlib import Path

import numpy as np
import pytest

# 8/sqrt(pi): the classical thermal enhancement of the Lorentz conductivity
# over its monoenergetic value at the thermal speed (derivation 2 above).
LORENTZ_THERMAL_FACTOR = 8.0 / sqrt(pi)


def _maxwellian_moment(m: float) -> float:
    """``(4/sqrt(pi)) int_0^inf x^m e^{-x^2} dx = (2/sqrt(pi)) Gamma((m+1)/2)``."""
    return (2.0 / sqrt(pi)) * gamma(0.5 * (m + 1.0))


# =============================================================================
# 1-3. Thermal parallel conductivity from the full drift-kinetic solve
# =============================================================================

DECK = """\
&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 1
  epsilon_t = {epsilon_t}d+0
  epsilon_h = 0d+0
  iota = {iota}d+0
  GHat = {g_hat}d+0
  IHat = {i_hat}d+0
  helicity_l = 2
  helicity_n = 10
  B0OverBBar = {b0}d+0
/
&speciesParameters
  Zs = {zs}
  mHats = {m_hats}
  nHats = {n_hats}
  THats = {t_hats}
  dNHatdpsiHats = {zeros}
  dTHatdpsiHats = {zeros}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = {nu_n}d+0
  EParallelHat = 1d+0
  dPhiHatdpsiHat = {dphi}d+0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .false.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {n_theta}
  Nzeta = 1
  Nxi = {n_xi}
  NL = 2
  Nx = {n_x}
  solverTolerance = 1d-13
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
&export_f
  export_full_f = .false.
  export_delta_f = .false.
/
"""

_DECK_DEFAULTS = {
    "epsilon_t": 0.0,
    "iota": 0.5,
    "g_hat": 3.0,
    "i_hat": 0.0,
    "b0": 1.0,
    "zs": "1",
    "m_hats": "1",
    "n_hats": "1.0d+0",
    "t_hats": "1.0d+0",
    "zeros": "0",
    "nu_n": 1.0,
    "dphi": 0.0,
    "n_theta": 5,
    "n_xi": 4,
    "n_x": 6,
}


def _write_deck(tmp_path: Path, name: str, **overrides: object) -> Path:
    fields = dict(_DECK_DEFAULTS)
    fields.update(overrides)
    path = tmp_path / name
    path.write_text(DECK.format(**fields))
    return path


def _solve_flow(deck: Path) -> tuple[float, float]:
    """Solve ``deck`` and return ``(FSABFlow[0], closed form (1) for species 0)``.

    The closed form is rebuilt here from the operator's own species data and
    speed grid and from ``nu_d_hat_pitch_angle_scattering_v3``; it references
    no geometry quantity at all (derivation 1).
    """
    from dkx.collisions import nu_d_hat_pitch_angle_scattering_v3
    from dkx.run import run_profile

    run = run_profile(deck, emit=None)
    op = run.operator
    x = np.asarray(op.x, dtype=float)
    w = np.asarray(op.x_weights, dtype=float)
    nu_d = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            x=op.x, z_s=op.z_s, m_hats=op.m_hat, n_hats=op.n_hat, t_hats=op.t_hat
        )
    )
    z = np.asarray(op.z_s, dtype=float)
    n = np.asarray(op.n_hat, dtype=float)
    m_hat = np.asarray(op.m_hat, dtype=float)
    speed_sum = float(np.sum(w * x**4 * np.exp(-x * x) / nu_d[0]))
    e_parallel_hat = 1.0  # every deck in this file sets EParallelHat = 1
    closed_form = (
        4.0
        * float(op.alpha)
        * z[0]
        * n[0]
        * e_parallel_hat
        / (3.0 * sqrt(pi) * m_hat[0] * float(op.pas.nu_n))
        * speed_sum
    )
    return float(np.asarray(run.moments["FSABFlow"])[0]), closed_form


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("base", {}),
        ("field-and-flux-functions", {"b0": 2.5, "iota": 1.3, "g_hat": 7.0, "i_hat": 1.7}),
        ("radial-electric-field", {"dphi": 0.4}),
        ("hotter-heavier", {"t_hats": "3.0d+0", "m_hats": "2.0d+0", "nu_n": 0.03}),
    ],
)
def test_inductive_response_matches_the_l1_closed_form(
    tmp_path: Path, label: str, overrides: dict
) -> None:
    """``FSABFlow`` equals closed form (1) exactly when ``|B|`` is constant.

    Derivation 1 of the module docstring: with ``epsilon_t = epsilon_h = 0``
    the geometryScheme-1 field is uniform, the theta/zeta-independent L=1
    response is annihilated by streaming and mirror terms, and the inductive
    drive is balanced by the pitch-angle eigenvalue alone, giving

        FSABFlow = 4 alpha Z n E_par / (3 sqrt(pi) mHat nu_n)
                   * sum_x w x^4 e^{-x^2} / nuDHat(x).

    The right-hand side contains no geometry, so the same value must come out
    of decks with different ``B0``, ``iota``, ``GHat``, ``IHat`` and with the
    ExB drift switched on -- which is the content of the parametrization.
    Measured relative deviation <= 2.3e-16 on all four; asserted 1e-12.
    """
    deck = _write_deck(tmp_path, f"uniform_b_{label}.namelist", **overrides)
    measured, closed_form = _solve_flow(deck)
    assert closed_form != 0.0
    rel = abs(measured - closed_form) / abs(closed_form)
    assert rel < 1e-12, f"{label}: measured={measured!r} closed_form={closed_form!r}"


def test_lorentz_gas_conductivity_is_the_classical_thermal_gamma_function(
    tmp_path: Path,
) -> None:
    """The thermal Lorentz conductivity and its ``8/sqrt(pi)`` enhancement.

    A Lorentz gas is built as two species: a test species of mass ``1e-12``
    and negligible density scattering off a unit-mass, unit-density
    background.  Then ``x_b = x sqrt(T_a m_b / (T_b m_a)) = 1e6 x``, so
    ``erf(x_b) - Psi(x_b) = 1 - O(1e-12)`` and the deflection frequency is the
    pure power law ``nuDHat = nuHat0 / x^3`` with

        nuHat0 = (3 sqrt(pi)/4) Z_a^2 sum_b Z_b^2 n_b / (THat_a^{3/2} sqrt(mHat_a))

    (read straight off ``nu_d_hat_pitch_angle_scattering_v3``).  The test
    species' own contribution to that sum -- the e-e term that Spitzer's
    momentum restoration acts on, and which is *excluded* from the Lorentz
    limit by definition -- is suppressed to 1e-14 by its density.

    The Gaussian speed grid then makes the speed sum in (1) exact,
    ``sum_x w x^7 e^{-x^2} = Gamma(4)/2 = 3``, and

        FSABFlow = 4 alpha Z n E_par / (sqrt(pi) mHat nu_n nuHat0),

    which is ``8/sqrt(pi) = 4.5135166683820502`` times the same quantity with
    the deflection frequency frozen at the thermal speed (``sum_x w x^4
    e^{-x^2} = 3 sqrt(pi)/8``).  Measured 1.6e-13 relative on both; asserted
    1e-10 (ratio measured 1.7e-13).  The residual is the finite mass ratio,
    not the solver: the same run reproduces the *exact*-``nuDHat`` form (1) to
    1.9e-16.
    """
    from dkx.collisions import nu_d_hat_pitch_angle_scattering_v3
    from dkx.run import run_profile

    deck = _write_deck(
        tmp_path,
        "lorentz_gas.namelist",
        zs="1 1",
        m_hats="1d-12 1",
        n_hats="1d-14 1.0d+0",
        t_hats="1.0d+0 1.0d+0",
        zeros="0 0",
    )
    run = run_profile(deck, emit=None)
    op = run.operator
    x = np.asarray(op.x, dtype=float)
    w = np.asarray(op.x_weights, dtype=float)
    z = np.asarray(op.z_s, dtype=float)
    n = np.asarray(op.n_hat, dtype=float)
    m_hat = np.asarray(op.m_hat, dtype=float)
    t_hat = np.asarray(op.t_hat, dtype=float)
    nu_n = float(op.pas.nu_n)
    alpha = float(op.alpha)

    # The deflection frequency really is the Lorentz power law.
    nu_d = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            x=op.x, z_s=op.z_s, m_hats=op.m_hat, n_hats=op.n_hat, t_hats=op.t_hat
        )
    )
    nu_hat0 = (
        0.75 * sqrt(pi) * z[0] ** 2 * float(np.sum(z * z * n)) / (t_hat[0] ** 1.5 * sqrt(m_hat[0]))
    )
    assert np.max(np.abs(nu_d[0] * x**3 / nu_hat0 - 1.0)) < 1e-10

    measured = float(np.asarray(run.moments["FSABFlow"])[0])

    # (2): the Gamma-function closed form.
    lorentz = 4.0 * alpha * z[0] * n[0] / (sqrt(pi) * m_hat[0] * nu_n * nu_hat0)
    assert abs(measured - lorentz) / abs(lorentz) < 1e-10

    # (3): the thermal enhancement over the monoenergetic value at x = 1,
    # i.e. over (1) evaluated with nuDHat frozen at nuHat0.
    monoenergetic = (
        4.0
        * alpha
        * z[0]
        * n[0]
        / (3.0 * sqrt(pi) * m_hat[0] * nu_n)
        * float(np.sum(w * x**4 * np.exp(-x * x))) / nu_hat0
    )
    ratio = measured / monoenergetic
    assert abs(ratio - LORENTZ_THERMAL_FACTOR) / LORENTZ_THERMAL_FACTOR < 1e-10, ratio


def test_high_collisionality_conductivity_approaches_the_geometry_free_limit(
    tmp_path: Path,
) -> None:
    """Pfirsch-Schlueter limit: trapping correction is O(eps_t^2/nu_n^2), negative.

    Closed form (1) holds exactly only for constant ``|B|``.  Switching on a
    tokamak modulation ``epsilon_t`` makes the solve fall *below* it, and
    derivation 3 of the module docstring fixes both the sign (the variational
    identity ``<S,A^{-1}S> <= <S,C^{-1}S>`` for antisymmetric streaming) and
    the two exponents (Legendre parity kills the first-order correction to the
    flow moment; each streaming application carries one ``grad_par BHat``).
    So the relative deviation must behave as ``-c epsilon_t^2 / nu_n^2``.

    Measured orders (2026-08-31): 1.9897 and 1.9989 in ``nu_n`` over
    ``10 -> 30 -> 100``; 1.9999 and 1.9995 in ``epsilon_t`` over
    ``0.05 -> 0.1 -> 0.2``.  Asserted within +-0.1 of 2.  Runtime ~10 s.
    """
    grid = {"n_theta": 9, "n_xi": 16}

    def deviation(epsilon_t: float, nu_n: float) -> float:
        deck = _write_deck(
            tmp_path,
            f"ps_e{epsilon_t}_n{nu_n}.namelist",
            epsilon_t=epsilon_t,
            nu_n=nu_n,
            **grid,
        )
        measured, closed_form = _solve_flow(deck)
        return (measured - closed_form) / closed_form

    nu_values = (10.0, 30.0, 100.0)
    nu_dev = [deviation(0.1, value) for value in nu_values]
    eps_values = (0.05, 0.1, 0.2)
    eps_dev = [deviation(value, 30.0) for value in eps_values]

    # Sign: streaming can only reduce the conductivity.
    for value in (*nu_dev, *eps_dev):
        assert value < 0.0, (nu_dev, eps_dev)

    # The geometry-free closed form is the actual high-collisionality limit.
    assert abs(nu_dev[-1]) < 1e-5, nu_dev

    def order(dev: list[float], knob: tuple[float, ...], i: int) -> float:
        """|d log|deviation| / d log knob| between consecutive scan points."""
        return abs(
            float(np.log(abs(dev[i + 1]) / abs(dev[i])) / np.log(knob[i + 1] / knob[i]))
        )

    for i in range(2):
        p = order(nu_dev, nu_values, i)
        assert 1.9 < p < 2.1, f"nu_n order {p} from {nu_dev}"
    for i in range(2):
        p = order(eps_dev, eps_values, i)
        assert 1.9 < p < 2.1, f"epsilon_t order {p} from {eps_dev}"


# =============================================================================
# 4. Monoenergetic-to-thermal convolution vs manufactured analytic kernels
# =============================================================================


# Deliberately non-round, sign-mixed geometry so that a dropped or duplicated
# factor in the un-normalization cannot cancel by accident.
CONV_GEOMETRY = {
    "delta": 4.5694e-3,
    "alpha": 1.0,
    "g_hat": 3.7481,
    "i_hat": 0.31,
    "iota": -0.4542,
    "b0": 1.07,
    "fsab2": 1.21,
    "r_hat": 0.19,
}

_F_CIRCULATING = 1.46  # Beidler's 1 - f_c = 1.46 sqrt(eps_t); monoenergetic._FT_LARGE_ASPECT


def _manufactured_database(**tables: object):
    """A :class:`MonoenergeticDatabase` with prescribed tables and geometry."""
    from dkx.monoenergetic import MonoenergeticDatabase

    return MonoenergeticDatabase(
        x0=1.0,
        w0=1.0,
        nu_d_hat_x0=1.0,
        delta=CONV_GEOMETRY["delta"],
        alpha=CONV_GEOMETRY["alpha"],
        g_hat=CONV_GEOMETRY["g_hat"],
        i_hat=CONV_GEOMETRY["i_hat"],
        iota=CONV_GEOMETRY["iota"],
        b0_over_bbar=CONV_GEOMETRY["b0"],
        fsab_hat2=CONV_GEOMETRY["fsab2"],
        r_hat=CONV_GEOMETRY["r_hat"],
        **tables,  # type: ignore[arg-type]
    )


def _reference_prefactors(z: float, m_hat: float, t_hat: float, nu_n: float) -> dict[str, float]:
    """Beidler reference values of ``monoenergetic.energy_convolution``.

    ``D11^p = K11 x^3``, ``D31^b = K31 x^2`` and ``D33^PS = K33 x^2/nuDHat``,
    written out independently of the module under test.
    """
    b0 = abs(CONV_GEOMETRY["b0"])
    iota = abs(CONV_GEOMETRY["iota"])
    r_major = abs(CONV_GEOMETRY["g_hat"]) / b0
    eps_t = CONV_GEOMETRY["r_hat"] / r_major
    vth = sqrt(t_hat / m_hat)
    v_d_over_x2 = CONV_GEOMETRY["delta"] * m_hat * vth * vth / (2.0 * z * r_major * b0)
    return {
        "k11": (pi / 4.0) * v_d_over_x2**2 * r_major / (iota * vth),
        "k31": (2.0 / 3.0)
        * (v_d_over_x2 * r_major / (iota * eps_t))
        * (_F_CIRCULATING * sqrt(eps_t)),
        "k33_over_nu": vth * vth * CONV_GEOMETRY["fsab2"] / (3.0 * nu_n * b0 * b0),
        "vth": vth,
        "r_major": r_major,
        "eps_t": eps_t,
    }


def test_convolution_of_power_law_coefficients_matches_gamma_closed_form() -> None:
    """All nine ``L_ij`` against Gamma functions for manufactured ``D*(nuPrime)``.

    The database is filled with pure power laws ``D11* = a11 nuPrime^{-1/2}``
    and ``D33* = a33 nuPrime^{1/2}`` (log-log interpolation of a power law is
    exact, so the lookup adds no error) and constants ``D13*``, ``D31*``.  The
    species set is the Lorentz gas of the conductivity test, pushed to a
    ``1e-16`` mass ratio (there is no solve here, so nothing limits it), so
    ``nuDHat = nuHat0/x^3`` and

        nuPrime(x) = (|GHat + iota IHat|/B0)(x0/nuDHat(x0)) nu_n nuDHat(x)/v
                   = C_nu x^{-4},      v = x vtHat,

    making every kernel a pure power of ``x``:

        d11 = a11 C_nu^{-1/2} K11 x^{3+2},  d13/d31 = b13/b31 K31 x^2,
        d33 = a33 C_nu^{+1/2} K33 x^{5-2},

    with ``K11``, ``K31``, ``K33`` the Beidler reference values written out in
    ``_reference_prefactors``.  With ``M(m) = (2/sqrt(pi)) Gamma((m+1)/2)`` the
    convolution is then, entry by entry,

        L = [[G11 K11 M(7),  G11 K11 M(9),  b13 K31 M(4)],
             [G11 K11 M(9),  G11 K11 M(11), b13 K31 M(6)],
             [b31 K31 M(4),  b31 K31 M(6),  G33 K33 M(5)]].

    Quadrature is the SFINCS speed grid with ``Nx = 12``, exact for every
    integrand here (highest is ``x^11 e^{-x^2}``, needing ``Nx >= 6``), so the
    comparison is exact arithmetic, not a converged approximation.  Measured
    max relative deviation 1.5e-15; asserted 1e-11.
    """
    from dkx.collisions import nu_d_hat_pitch_angle_scattering_v3
    from dkx.monoenergetic import energy_convolution
    from dkx.xgrid import make_x_grid

    z_s = np.array([1.0, 1.0])
    m_hats = np.array([1.0e-16, 1.0])
    n_hats = np.array([0.0, 1.0])  # n enters ONLY nuDHat: zero removes self-collisions
    t_hats = np.array([1.0, 1.0])
    nu_n = 0.7

    grid = make_x_grid(n=12, k=0.0)
    x = grid.x
    w = grid.dx_weights()
    nu_d = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            x=x, z_s=z_s, m_hats=m_hats, n_hats=n_hats, t_hats=t_hats
        )
    )
    nu_hat0 = (
        0.75
        * sqrt(pi)
        * z_s[0] ** 2
        * float(np.sum(z_s * z_s * n_hats))
        / (t_hats[0] ** 1.5 * sqrt(m_hats[0]))
    )
    assert np.max(np.abs(nu_d[0] * x**3 / nu_hat0 - 1.0)) < 1e-12

    a11, p11 = 2.5, -0.5
    a33, p33 = 0.8, 0.5
    b13, b31 = 0.35, -0.42
    nu_grid = np.geomspace(1e-24, 1e24, 5)
    e_star_grid = np.array([0.0, 1.0])
    flat = np.ones((nu_grid.size, e_star_grid.size))
    db = _manufactured_database(
        nu_prime=nu_grid,
        e_star=e_star_grid,
        d11_star=(a11 * nu_grid**p11)[:, None] * flat,
        d13_star=b13 * flat,
        d31_star=b31 * flat,
        d33_star=(a33 * nu_grid**p33)[:, None] * flat,
    )

    thermal = energy_convolution(
        db, z_s=z_s, m_hats=m_hats, t_hats=t_hats, n_hats=n_hats, nu_n=nu_n, x=x, x_weights=w
    )
    measured = np.asarray(thermal.l_matrix[0])

    ref = _reference_prefactors(z_s[0], m_hats[0], t_hats[0], nu_n)
    g_plus = abs(CONV_GEOMETRY["g_hat"] + CONV_GEOMETRY["iota"] * CONV_GEOMETRY["i_hat"])
    c_nu = g_plus / abs(CONV_GEOMETRY["b0"]) * nu_n * nu_hat0 / ref["vth"]  # x0/nuDHat(x0) = 1
    g11 = a11 * c_nu**p11
    g33 = a33 * c_nu**p33
    k11, k31 = ref["k11"], ref["k31"]
    k33 = ref["k33_over_nu"] / nu_hat0
    q = 3.0 - 4.0 * p11  # exponent of x in d11
    t = 5.0 - 4.0 * p33  # exponent of x in d33
    moment = _maxwellian_moment
    expected = np.array(
        [
            [g11 * k11 * moment(2 + q), g11 * k11 * moment(4 + q), b13 * k31 * moment(4)],
            [g11 * k11 * moment(4 + q), g11 * k11 * moment(6 + q), b13 * k31 * moment(6)],
            [b31 * k31 * moment(4), b31 * k31 * moment(6), g33 * k33 * moment(2 + t)],
        ]
    )
    rel = np.abs(measured - expected) / np.abs(expected)
    assert rel.max() < 1e-11, (measured, expected, rel)

    # The claimed speed exponents, isolated: these ratios are independent of
    # every common prefactor, so only M(m)'s argument can make them agree.
    assert measured[0, 1] / measured[0, 0] == pytest.approx(
        moment(4 + q) / moment(2 + q), rel=1e-11
    )
    assert measured[2, 1] / measured[2, 0] == pytest.approx(moment(6) / moment(4), rel=1e-11)


def test_convolution_of_constant_coefficients_is_deflection_frequency_free() -> None:
    """Eight of nine ``L_ij`` in closed form for *any* deflection frequency.

    With the ``D*`` tables held constant the ``nuPrime`` lookup drops out and
    the speed dependence of the kernels comes entirely from the Beidler
    reference values, ``D11^p prop x^3`` and ``D31^b prop x^2`` -- neither of
    which involves the collision frequency.  So for a realistic hydrogen and
    carbon mixture with the code's own ``nuDHat``,

        L_11 = a11 K11 M(5),  L_12 = L_21 = a11 K11 M(7),  L_22 = a11 K11 M(9),
        L_13 = b13 K31 M(4),  L_23 = b13 K31 M(6),
        L_31 = b31 K31 M(4),  L_32 = b31 K31 M(6),

    with ``M(5) = 4/sqrt(pi)``, ``M(7) = 12/sqrt(pi)``, ``M(9) = 48/sqrt(pi)``,
    ``M(4) = 3/2`` and ``M(6) = 15/4``.  ``L_33`` is the one entry that carries
    ``1/nuDHat(x)``; it is checked against the same integral written out
    directly.  This complements the power-law test: there the collision
    frequency was forced into the Lorentz form, here it is arbitrary.
    Measured max relative deviation 8.0e-16 over both species; asserted 1e-12.
    """
    from dkx.collisions import nu_d_hat_pitch_angle_scattering_v3
    from dkx.monoenergetic import energy_convolution
    from dkx.xgrid import make_x_grid

    z_s = np.array([1.0, 6.0])
    m_hats = np.array([1.0, 12.0])
    n_hats = np.array([0.6, 0.04])
    t_hats = np.array([0.8, 1.1])
    nu_n = 0.013

    grid = make_x_grid(n=12, k=0.0)
    x = grid.x
    w = grid.dx_weights()
    nu_d = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            x=x, z_s=z_s, m_hats=m_hats, n_hats=n_hats, t_hats=t_hats
        )
    )
    # Not a power law: this is the point of the test.
    assert np.ptp(nu_d[0] * x**3) / np.mean(nu_d[0] * x**3) > 0.5

    a11, b13, b31, a33 = 1.7, 0.35, -0.42, 0.8
    nu_grid = np.geomspace(1e-6, 1e6, 3)
    e_star_grid = np.array([0.0, 1.0])
    flat = np.ones((nu_grid.size, e_star_grid.size))
    db = _manufactured_database(
        nu_prime=nu_grid,
        e_star=e_star_grid,
        d11_star=a11 * flat,
        d13_star=b13 * flat,
        d31_star=b31 * flat,
        d33_star=a33 * flat,
    )
    thermal = energy_convolution(
        db, z_s=z_s, m_hats=m_hats, t_hats=t_hats, n_hats=n_hats, nu_n=nu_n, x=x, x_weights=w
    )

    moment = _maxwellian_moment
    assert moment(5) == pytest.approx(4.0 / sqrt(pi), rel=1e-14)
    assert moment(7) == pytest.approx(12.0 / sqrt(pi), rel=1e-14)
    assert moment(9) == pytest.approx(48.0 / sqrt(pi), rel=1e-14)
    assert moment(4) == pytest.approx(1.5, rel=1e-14)
    assert moment(6) == pytest.approx(3.75, rel=1e-14)

    for s in range(z_s.size):
        measured = np.asarray(thermal.l_matrix[s])
        ref = _reference_prefactors(z_s[s], m_hats[s], t_hats[s], nu_n)
        k11, k31 = ref["k11"], ref["k31"]
        expected = np.array(
            [
                [a11 * k11 * moment(5), a11 * k11 * moment(7), b13 * k31 * moment(4)],
                [a11 * k11 * moment(7), a11 * k11 * moment(9), b13 * k31 * moment(6)],
                [b31 * k31 * moment(4), b31 * k31 * moment(6), np.nan],
            ]
        )
        rel = np.abs(measured - expected) / np.abs(expected)
        assert np.nanmax(rel) < 1e-12, (s, measured, expected, rel)

        # L_33 is the only entry that sees the deflection frequency.
        l33 = ref["k33_over_nu"] * (4.0 / sqrt(pi)) * float(
            np.sum(w * x**4 * np.exp(-x * x) / nu_d[s])
        )
        assert measured[2, 2] == pytest.approx(a33 * l33, rel=1e-12)


def test_convolved_lorentz_conductivity_shows_the_same_thermal_enhancement() -> None:
    """The convolution route reproduces ``8/sqrt(pi)`` too (cross-route check).

    Setting ``D33* = 1`` -- its exact collisional value, already pinned by
    ``test_monoenergetic_database.py::test_normalization_physics_gates`` -- and
    using the Lorentz deflection frequency makes ``d33 = K33 x^5``, so the
    convolved ``L_33`` equals ``K33 M(7) = 12 K33/sqrt(pi)``.  Freezing only
    the deflection frequency at the thermal speed leaves ``d33 = K33 x^2`` and
    ``L_33 = K33 M(4) = 3 K33/2``, so the ratio is

        M(7)/M(4) = (12/sqrt(pi)) / (3/2) = 8/sqrt(pi),

    exactly the enhancement of derivation 2, reached through the monoenergetic
    database instead of through a drift-kinetic solve.  Measured 2.2e-15;
    asserted 1e-12.
    """
    from dkx.monoenergetic import energy_convolution
    from dkx.xgrid import make_x_grid

    z_s = np.array([1.0, 1.0])
    m_hats = np.array([1.0e-16, 1.0])
    n_hats = np.array([0.0, 1.0])
    t_hats = np.array([1.0, 1.0])
    nu_n = 0.7

    grid = make_x_grid(n=12, k=0.0)
    x = grid.x
    w = grid.dx_weights()
    nu_hat0 = 0.75 * sqrt(pi) / (t_hats[0] ** 1.5 * sqrt(m_hats[0]))

    nu_grid = np.geomspace(1e-24, 1e24, 5)
    e_star_grid = np.array([0.0, 1.0])
    flat = np.ones((nu_grid.size, e_star_grid.size))
    db = _manufactured_database(
        nu_prime=nu_grid,
        e_star=e_star_grid,
        d11_star=flat,
        d13_star=0.0 * flat,
        d31_star=0.0 * flat,
        d33_star=flat,
    )
    thermal = energy_convolution(
        db, z_s=z_s, m_hats=m_hats, t_hats=t_hats, n_hats=n_hats, nu_n=nu_n, x=x, x_weights=w
    )
    l33 = float(np.asarray(thermal.l_matrix[0])[2, 2])

    ref = _reference_prefactors(z_s[0], m_hats[0], t_hats[0], nu_n)
    k33 = ref["k33_over_nu"] / nu_hat0
    assert l33 == pytest.approx(k33 * _maxwellian_moment(7), rel=1e-12)

    # Freezing nuDHat at the thermal speed leaves d33 = K33 x^2 -> K33 M(4).
    frozen = k33 * _maxwellian_moment(4)
    assert l33 / frozen == pytest.approx(LORENTZ_THERMAL_FACTOR, rel=1e-12)
