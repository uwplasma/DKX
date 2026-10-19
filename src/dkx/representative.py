"""One-command representative run: an equilibrium in, publication panels out.

``dkx wout_XXX.nc`` solves the small set of cases a neoclassical paper actually
reports and renders them as one figure, in about a minute rather than hours.
``dkx --plot FILE`` renders the same panels from an existing output file --- and
because :mod:`dkx.writer` emits ``sfincsOutput.h5`` in SFINCS's own layout, that
works on Fortran SFINCS output too, with no separate reader.

Panels
------
* **monoenergetic** ``D11*``, ``D31*``, ``D33*`` against ``nuPrime``, one curve
  per ``EStar``.  The standard cross-code benchmark figure.
* **bootstrap** ``<j.B>/sqrt(<B^2>)`` in kA/m^2 against radius, evaluated at
  the ambipolar root or, when the fixed scan does not bracket one, the sampled
  ``E_r`` with smallest ``|J_r|``.  The latter is visibly marked as a sampled
  point, not a root.  The VMEC equilibrium's own current is drawn beside it.
* **fluxes** ``<Gamma.grad r>`` and ``<Q.grad r>`` per species against radius,
  in SI units (:mod:`dkx.units`), at that same explicitly labeled field.
* **|B|** on the flux surface, for context on what device produced the numbers.

Resolution and benchmark rationale live with the maintained user guidance in
:doc:`usage` and :doc:`performance`.  Monoenergetic runs take
``nuPrime``/``EStar``, not ``nu_n``/``Er``.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .representative_plots import (  # noqa: F401
    _import_matplotlib,
    _panel_ambipolarity,
    _panel_bootstrap,
    _panel_fluxes,
    _panel_modB,
    _panel_monoenergetic,
    _panel_radial_bootstrap,
    _panel_radial_fluxes,
    _panel_run_summary,
    _panel_vs_x,
    _profile_panel_absence,
    _scalar,
    _species_labels,
    figure_caption,
    plot_output_file,
    plot_representative,
    plot_single_run,
)

#: Convergence-tested default, see the module docstring.
#: ``n_xi`` must be at least ``n_zeta``: at low collisionality the pitch-angle
#: resolution is what limits the answer, and the convergence scan that set the
#: earlier 25/41/20 was run at a single mid collisionality, where it does not
#: show.  25/25/41 on the collaborator's advice from a full-range test.
DEFAULT_RESOLUTION = {"n_theta": 25, "n_zeta": 25, "n_xi": 41}

#: Resolution for the single RHSMode=1 solve behind the bootstrap/flux panels.
#: Smaller than the monoenergetic grid because it runs once, not 21 times.
DEFAULT_RESOLUTION_PROFILE = {"n_theta": 13, "n_zeta": 19, "n_xi": 13, "n_x": 4}

#: ``--full`` resolution for the same solves.  The two are a measured trade, not
#: a guess.  On the precise-QA finite-beta reference at ``r/a = 0.25`` with the
#: Fokker-Planck operator, against a ``21x31x24x5`` reference:
#:
#: ===================  =========  ==========  =================  ==========
#: grid                 unknowns   s / solve   ``<j.B>`` error    root error
#: ===================  =========  ==========  =================  ==========
#: ``21x31x24x5``       156,240    25.3        reference          reference
#: ``15x23x16x5``        55,200     9.26       0.1%               5%
#: ``13x19x13x4``        25,688     1.00       2.8%               24%
#: ===================  =========  ==========  =================  ==========
#:
#: The default takes the last row because **the root is the sensitive quantity
#: and nothing reported at it is**: ``<j.B>`` moves 0.6% between ``E_r = -3``
#: and ``-6`` kV/m, so a 24% error in the root costs under 1% in the current and
#: the fluxes drawn there.  ``--full`` takes the middle row when the ambipolar
#: field itself is the number wanted.  The 25x jump between the top two rows is
#: a solver-route change, not smooth scaling, which is why the default sits
#: below it rather than halfway.
FULL_RESOLUTION_PROFILE = {"n_theta": 15, "n_zeta": 23, "n_xi": 16, "n_x": 5}

#: ``--quick`` resolution and scan sizes: the smallest run that still exercises
#: every stage of :func:`run_representative`.
#:
#: This is a **smoke preset, not a physics one**.  It exists because the
#: default run costs 64.9 s cold-cache on a 10-core M4 against 16.1 s for this
#: one (both on ``tests/ref/wout_up_down_asymmetric_tokamak.nc``) --- fine for
#: a person either way, but the CI job that runs it has to build a wheel and
#: install it first --- and because the failure that job exists to catch (a
#: file the package needs that the wheel does not ship) happens while the deck
#: is being built, long before the grid matters.  Do not report numbers from
#: it:
#:
#: * ``QUICK_NU_PRIME`` starts at 1e-2, so it misses the ``1/nu`` branch that
#:   :data:`DEFAULT_NU_PRIME` exists to show;
#: * ``QUICK_ER_BRACKET`` stops at -8 kV/m, so a device whose ion root sits
#:   below that reports no root at all rather than a wrong one;
#: * every angular axis is below the convergence floor in the module docstring.
#:
#: What it does keep is the shape of the run: an RHSMode=3 scan with more than
#: one ``EStar`` curve and more than two ``nuPrime`` points per curve, one
#: RHSMode=1 profile solve, and a batched ``E_r`` scan over more than one
#: surface.  Cutting any of those to one point turns a panel into a dot and
#: stops the corresponding code path from running at all.
#:
#: ``n_x`` is the exception that is **not** cut, and the measurement is worth
#: recording.  At ``n_x = 3`` the ``E_r`` scan on the reference tokamak returns
#: a radial current that is negative across the whole bracket: no sign change,
#: no ambipolar root, and the bootstrap and flux panels come out empty.  At
#: ``n_x = 4``, everything else unchanged, the same run finds roots at -1.25
#: and -1.80 kV/m, and the grid below finds -1.57 and -2.36 against the default
#: grid's -1.56 and -2.34.  The speed grid is what the root is sensitive to, so
#: it stays at the default's 4 while the angular axes go to roughly half.
QUICK_RESOLUTION = {"n_theta": 11, "n_zeta": 13, "n_xi": 16}
QUICK_RESOLUTION_PROFILE = {"n_theta": 11, "n_zeta": 15, "n_xi": 10, "n_x": 4}
QUICK_NU_PRIME = (1.0e-2, 1.0e0, 1.0e2)
QUICK_E_STAR = (0.0, 1.0e-1)
QUICK_SURFACES = (0.4, 0.7)
QUICK_ER_BRACKET = (-8.0, -4.0, -2.0, -1.0, -0.4, 0.4, 1.0)

#: Generic fallback plasma, used only when the equilibrium carries no pressure.
FALLBACK_PLASMA = {"n_hat": 1.0, "t_hat": 1.0, "dn_drhat": -0.5, "dt_drhat": -1.0}

#: A deuterium/electron pair at modest collisionality, with the density and
#: temperature gradients that make the bootstrap current nonzero.  The gradient
#: keys must match ``inputRadialCoordinateForGradients``, and that key lives in
#: ``&geometryParameters``: a mismatch leaves the gradients at ZERO and the
#: whole solve returns ~1e-20 -- a run that completes and drives nothing.  The
#: template leaves it at the v3 default of 4, as the upstream decks do, because
#: **4 is the only code that drives the potential with ``Er``** --- any other
#: choice raises "Er != 0 with a non-Er inputRadialCoordinateForGradients", and
#: the ambipolar scan is an ``Er`` scan.  Code 4 takes n and T gradients with
#: respect to ``rHat``, so :func:`plasma_parameters` owes the chain rule
#: ``d/drHat = (1/aHat) d/drN``; ``aHat`` is 0.17 on a compact device, so
#: skipping it understates the drive six-fold.
#:
#: ``collisionOperator = 0`` is the full linearized Fokker-Planck operator.
#: Pitch-angle scattering is cheaper and fine for ``D11``, but the bootstrap
#: current is the parallel-momentum moment and PAS has no momentum-restoring
#: term: measured against Redl on a finite-beta precise-QA equilibrium it runs
#: 35-47% high, where Fokker-Planck lands within 2-7%
#: (:data:`dkx.bootstrap.DEFAULT_COLLISION_OPERATOR`).
#: The monoenergetic base for :func:`run_representative`.
#:
#: This is a module-level string on purpose.  It used to be read from
#: ``dkx/data/representative.namelist``, falling back to a deck under
#: ``examples/``, and **neither ships in the wheel** -- so ``dkx wout_*.nc``
#: worked in a source checkout and failed for every pip user.  It failed twice
#: over: FileNotFoundError when nothing was found, and, when some other
#: namelist happened to sit at one of those paths, a base carrying the default
#: ``RHSMode = 1`` that only surfaced as "run_transport_matrix supports RHSMode
#: 2 and 3" from three frames deeper.
#:
#: ``RHSMode = 3`` is the whole point of this deck; :func:`monoenergetic_scan`
#: checks it rather than trusting it.  ``Nx = 1`` is not a resolution to raise:
#: monoenergetic coefficients are defined at a single speed.
_MONOENERGETIC_TEMPLATE = """&general
  RHSMode = 3
/
&geometryParameters
  geometryScheme = 5
  equilibriumFile = "{equilibrium}"
  inputRadialCoordinate = 3
  rN_wish = 0.5
  VMECRadialOption = 1
  min_Bmn_to_load = 0
/
&speciesParameters
/
&physicsParameters
  nuPrime = 1.0d+0
  EStar = 0.2d+0
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
  solverTolerance = 1d-6
/
&otherNumericalParameters
/
&preconditionerOptions
/
"""

_PROFILE_TEMPLATE = """&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 5
  equilibriumFile = "{equilibrium}"
  VMECRadialOption = 0
  inputRadialCoordinate = 3
  rN_wish = 0.5
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = {n_hat:.6g} {n_hat:.6g}
  THats = {ti_hat:.6g} {te_hat:.6g}
  dNHatdrHats = {dn_drhat:.6g} {dn_drhat:.6g}
  dTHatdrHats = {dti_drhat:.6g} {dte_drhat:.6g}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 8.4774d-3
  Er = 0.0d+0
  collisionOperator = {collision_operator}
/
&resolutionParameters
  Ntheta = {n_theta}
  Nzeta = {n_zeta}
  Nxi = {n_xi}
  NL = 4
  Nx = {n_x}
  solverTolerance = 1d-8
/
&otherNumericalParameters
/
&preconditionerOptions
/
"""

#: Monoenergetic scan grid, spanning the three collisionality regimes a
#: neoclassical paper reports: the ``1/nu`` branch at low collisionality, the
#: plateau, and Pfirsch-Schlueter at high.  Two decades show none of that -- the
#: upstream decks themselves sit at ``nuPrime`` 1.2e-3 to 1.0, so a grid that
#: starts at 1e-2 misses the physics that makes a stellarator interesting.
DEFAULT_NU_PRIME = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1, 1.0e2)
#: ``EStar`` values.  The intermediate point belongs at 1e-3, not 0.1: between
#: zero field and 0.1 the D11 curves are nearly indistinguishable, so a grid of
#: 0/0.1/0.3 spends two of its three curves in the same regime.
DEFAULT_E_STAR = (0.0, 1.0e-3, 1.0e-1)


def _profile_resolution(*, full: bool = False, quick: bool = False) -> dict[str, int]:
    """The RHSMode=1 grid for one preset, and the one place the presets collide.

    ``full`` and ``quick`` pull in opposite directions, so taking both is a
    caller mistake rather than something to resolve silently.  Raising here
    rather than in :func:`run_representative` covers the two public stage
    functions as well, which a caller can drive directly.
    """
    if full and quick:
        raise ValueError("full and quick are opposite presets; pass at most one.")
    if quick:
        return QUICK_RESOLUTION_PROFILE
    return FULL_RESOLUTION_PROFILE if full else DEFAULT_RESOLUTION_PROFILE


def _quiet(fn: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def monoenergetic_scan(
    namelist: Any,
    *,
    nu_prime: Sequence[float] = DEFAULT_NU_PRIME,
    e_star: Sequence[float] = DEFAULT_E_STAR,
    resolution: dict[str, int] | None = None,
    emit: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Solve the ``(nuPrime, EStar)`` grid and return one record per point.

    Each record carries the full 2x2 transport matrix, so a caller can plot
    ``D11``/``D31``/``D33`` without re-running anything.
    """
    from dkx.inputs import read_sfincs_input, sfincs_input_from_raw  # noqa: PLC0415
    from dkx.run import run_transport_matrix  # noqa: PLC0415

    base = namelist
    if not hasattr(base, "resolution"):
        base = sfincs_input_from_raw(read_sfincs_input(namelist))
    if int(base.general.rhs_mode) != 3:
        raise ValueError(
            "monoenergetic_scan needs an RHSMode=3 deck; got "
            f"RHSMode={int(base.general.rhs_mode)}.  Passing an RHSMode=1 deck "
            "surfaces three frames deeper as 'run_transport_matrix supports "
            "RHSMode 2 and 3', which does not point back here."
        )
    res = dict(DEFAULT_RESOLUTION if resolution is None else resolution)

    records: list[dict[str, Any]] = []
    for e in e_star:
        for nu in nu_prime:
            inp = dataclasses.replace(
                base,
                resolution=dataclasses.replace(base.resolution, **res),
                physics=dataclasses.replace(
                    base.physics, nu_prime=float(nu), e_star=float(e)
                ),
            )
            t0 = time.perf_counter()
            run = _quiet(lambda: run_transport_matrix(inp, out_path=None, emit=None))
            matrix = np.asarray(run.transport_matrix)
            records.append({
                "nu_prime": float(nu), "e_star": float(e),
                "transport_matrix": matrix.tolist(),
                "D11": float(matrix[0, 0]), "D31": float(matrix[0, 1]),
                "D33": float(matrix[1, 1]),
                "wall_s": round(time.perf_counter() - t0, 3),
                "converged": bool(run.solve_result.converged),
            })  # fmt: skip
            if emit:
                emit(f"    nuPrime={nu:<8.3g} EStar={e:<5.3g} D11={matrix[0, 0]:+.4e}"
                     f"  ({records[-1]['wall_s']:.2f}s)")  # fmt: skip
    return records


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def run_representative(
    equilibrium: str | Path,
    *,
    out_path: str | Path | None = None,
    full: bool = False,
    quick: bool = False,
    emit: Callable[[str], None] | None = print,
    surfaces: Sequence[float] | None = None,
    er: float | None = None,
    collision_operator: str = "fp",
    profiles: Any = None,
    redl_jdotb: tuple[Sequence[float], Sequence[float]] | None = None,
) -> Path:
    """Solve the representative set for one equilibrium and plot it.

    Options (defaults reproduce ``dkx wout.nc``):

    - ``surfaces``: ``r/a`` values of the radial scan.
    - ``er``: ``None`` searches each surface for the ambipolar root; a float
      prescribes ``E_r`` in kV/m.
    - ``collision_operator``: ``"fp"``, ``"pas"`` or ``"pas+momentum_correction"``.
    - ``profiles``: explicit ``ne_coeffs`` [m^-3], ``Te_coeffs``/``Ti_coeffs``
      [eV], polynomials in ``s`` lowest order first (VMEX ``KineticProfiles``
      or a mapping), replacing the pressure split (:func:`profile_plasma`).
    - ``redl_jdotb``: ``(s, <j.B>)`` in A T/m^2 from an external Redl model,
      drawn beside the kinetic current (DKX has no Redl implementation).

    ``full`` widens the monoenergetic grid; the default keeps the whole run
    inside the one-minute budget the module docstring measures.  ``quick``
    goes the other way, to :data:`QUICK_RESOLUTION` and friends --- a smoke
    preset whose numbers are not reportable; see that constant for what it
    gives up.  The two are mutually exclusive.
    """
    from dkx.inputs import (  # noqa: PLC0415
        parse_sfincs_input_text,
        sfincs_input_from_raw,
    )

    equilibrium = Path(equilibrium)
    if not equilibrium.is_file():
        raise FileNotFoundError(f"no such equilibrium file: {equilibrium}")
    base = sfincs_input_from_raw(
        parse_sfincs_input_text(
            _MONOENERGETIC_TEMPLATE.format(equilibrium=str(equilibrium))
        )
    )
    # Each stage is timed and the time is printed.  The three stages differ by
    # more than an order of magnitude in cost, and on a large device the radial
    # scan dominates, so a reader deciding whether to trim DEFAULT_SURFACES
    # needs the split rather than one total at the end.
    started = time.perf_counter()

    def stage(label: str) -> Callable[[], None]:
        mark = time.perf_counter()
        if emit:
            emit(label)
        return lambda: (
            emit(f"    ({time.perf_counter() - mark:.1f} s)") if emit else None
        )

    profile_resolution = _profile_resolution(full=full, quick=quick)
    mono_resolution = QUICK_RESOLUTION if quick else DEFAULT_RESOLUTION
    done = stage(f"  monoenergetic scan at {mono_resolution}")
    if quick:
        nu, e_star = QUICK_NU_PRIME, QUICK_E_STAR
    else:
        nu = DEFAULT_NU_PRIME if not full else tuple(np.logspace(-5, 2, 15))
        e_star = DEFAULT_E_STAR
    scan = monoenergetic_scan(
        base, nu_prime=nu, e_star=e_star, resolution=mono_resolution, emit=emit
    )
    done()

    # The monoenergetic scan alone leaves the whole second row empty: RHSMode=3
    # produces a transport matrix and nothing else.  |B| comes free from the
    # geometry, and one RHSMode=1 solve on the same equilibrium supplies the
    # bootstrap current and the species fluxes.  Without this the figure renders
    # three "not present in this output" boxes, which is not a representative
    # run of anything.
    done = stage("  profile solve for |B|")
    plasma, plasma_source = resolve_plasma(equilibrium)
    if profiles is not None:
        a_hat = equilibrium_scalars(equilibrium)["a_hat"]
        plasma = profile_plasma(profiles, 0.5, a_hat)
        plasma_source = "explicit n, T profiles (polynomials in s)"
    data = _profile_data(equilibrium, full=full, quick=quick, emit=emit,
                         plasma=(plasma, plasma_source),
                         collision_operator=collision_operator)  # fmt: skip
    done()
    done = stage("  radial scan: ambipolar Er, bootstrap and fluxes at the root")
    kinetic = profiles
    profiles = radial_profiles(equilibrium, surfaces=surfaces, full=full, quick=quick,
                               emit=emit, profiles=kinetic, er=er,
                               collision_operator=collision_operator)  # fmt: skip
    done()
    if redl_jdotb is not None:
        s_redl, j_redl = (np.asarray(v, dtype=float) for v in redl_jdotb)
        for p in profiles:
            if p.get("root_fsab2"):
                p["jdotb_redl_kA_m2"] = float(
                    np.interp(p["r"] ** 2, s_redl, j_redl)) / p["root_fsab2"] / 1.0e3

    out = Path(out_path) if out_path else Path(f"{equilibrium.stem}.panels.png")
    figure = plot_representative(
        out, data=data, scan=scan, profiles=profiles,
        ambipolar=profiles if er is None and any(p.get("er_scan") for p in profiles) else None,
        plasma=plasma, plasma_source=plasma_source,
        resolutions={"monoenergetic": mono_resolution,
                     "profiles": profile_resolution},
        title=f"DKX representative run — {equilibrium.name}",
    )  # fmt: skip
    # Always leave the numbers behind, not only the picture: a figure cannot be
    # re-plotted, re-scaled or checked against another code.
    written = write_representative_output(
        out.with_suffix(".h5"), scan=scan, profiles=profiles, data=data,
        equilibrium=equilibrium,
        resolution_profile=profile_resolution,
        resolution_monoenergetic=mono_resolution,
    )  # fmt: skip
    if emit:
        emit(f" wrote {written}")
        emit(f" total {time.perf_counter() - started:.1f} s")
    return figure


#: Reference temperature (keV) used to split the equilibrium's pressure into a
#: density and a temperature.  p = n_i T_i + n_e T_e with quasineutrality and
#: T_i = T_e = T leaves p = 2 n T: one equation, two unknowns.  Fixing T on axis
#: and letting n carry the profile is the conventional choice, and it is stated
#: on the figure because it is an assumption, not a measurement.
#: Reference on-axis density and pressure, from the published kinetic profiles
#: of the reactor-scale QA and QH configurations in Landreman, Buller and
#: Drevlak, arXiv:2205.02914 -- the same study VMEX benchmarks its
#: self-consistent bootstrap current against:
#:
#:   QA: n_e(0) = 2.38e20 m^-3, T_e(0) = T_i(0) = 9.45 keV  -> p(0) = 7.207e5 Pa
#:   QH: n_e(0) = 2.20e20 m^-3, T_e(0) = T_i(0) = 10.0 keV  -> p(0) = 7.050e5 Pa
#:
#: with Z_eff = 1 and p = e(n_e T_e + n_i T_i) exactly. The pair below is their
#: mean. A hardcoded temperature was the previous approach and it was wrong by a
#: factor of five on exactly this configuration family: 2 keV against the
#: published 9.45, which put the density at 1.1e21 m^-3 and suppressed the
#: bootstrap current by ~2.5x through the n/T^2 collisionality.
REFERENCE_DENSITY_M3 = 2.29e20
REFERENCE_PRESSURE_PA = 7.1282e5

#: How the on-axis density scales with the equilibrium's own on-axis pressure:
#: ``n(0) = n_ref (p(0)/p_ref)^AXIS_DENSITY_PRESSURE_EXPONENT``, and the
#: temperature then *follows* from ``p = 2 n T`` rather than being assumed.
#:
#: One half splits the pressure equally between density and temperature in the
#: logarithm, which is the assumption-light choice: it is symmetric in the two
#: quantities, and it degrades gracefully far from the anchor where a steeper or
#: flatter exponent would not. Near the anchor the choice barely matters -- 1/2,
#: 2/3 and 1 give 9.77, 9.75 and 9.71 keV on the QA deck -- so this is chosen for
#: its behaviour away from the reference, not at it.
AXIS_DENSITY_PRESSURE_EXPONENT = 0.5


def axis_density_m3(p_axis_pa: float) -> float:
    """On-axis density implied by the equilibrium's own on-axis pressure."""
    if not p_axis_pa > 0.0:
        return REFERENCE_DENSITY_M3
    ratio = float(p_axis_pa) / REFERENCE_PRESSURE_PA
    return REFERENCE_DENSITY_M3 * ratio**AXIS_DENSITY_PRESSURE_EXPONENT


def axis_temperature_kev(p_axis_pa: float, n_axis_m3: float | None = None) -> float:
    """On-axis temperature from ``p = 2 n T`` -- derived, never assumed."""
    from dkx.units import ELEMENTARY_CHARGE  # noqa: PLC0415

    n0 = axis_density_m3(p_axis_pa) if n_axis_m3 is None else float(n_axis_m3)
    if not (n0 > 0.0 and p_axis_pa > 0.0):
        return 0.0
    return float(p_axis_pa) / (2.0 * ELEMENTARY_CHARGE * n0) / 1.0e3


#: Exponent splitting the pressure between temperature and density:
#: ``T ~ p^TEMPERATURE_PRESSURE_EXPONENT`` and therefore ``n ~ p^(1-e)``.
#: One third gives ``n ~ p^(2/3)``, the conventional mild split.  The exponent
#: form is what keeps the closure well behaved: both profiles then fall wherever
#: the pressure falls.  A temperature falling *linearly* against a flat-topped
#: pressure makes ``n = p/(2T)`` rise off-axis, i.e. invents a hollow density.
TEMPERATURE_PRESSURE_EXPONENT = 1.0 / 3.0


#: Explicit on-axis density override, in m^-3, or None to derive it from the
#: equilibrium's own pressure. Set by ``dkx --density-m3``; not an environment
#: variable, because a run's plasma should be visible in the command that
#: produced it rather than in the shell that happened to launch it.
_DENSITY_OVERRIDE_M3: float | None = None


def set_axis_density_override(value: float | None) -> None:
    """Pin the on-axis density, or pass ``None`` to derive it from the pressure."""
    global _DENSITY_OVERRIDE_M3
    if value is not None and not float(value) > 0.0:
        raise ValueError(f"on-axis density must be positive, got {value!r}")
    _DENSITY_OVERRIDE_M3 = None if value is None else float(value)


def _n_axis_m3(p_axis: float) -> float:
    """The on-axis density this run is using: the override, else from pressure."""
    if _DENSITY_OVERRIDE_M3 is not None:
        return _DENSITY_OVERRIDE_M3
    return axis_density_m3(p_axis)


def _t_axis_kev(p_axis: float) -> float:
    """The on-axis temperature, always derived from ``p = 2 n T``."""
    return axis_temperature_kev(p_axis, _n_axis_m3(p_axis))


def _temperature(p_pa: float, p_axis: float) -> float:
    """``T(s) = T_0 (p(s)/p(0))^e`` in keV --- the assumed half of the closure."""
    t_axis = _t_axis_kev(p_axis)
    if p_axis <= 0.0:
        return t_axis
    ratio = max(float(p_pa) / float(p_axis), 0.0)
    return t_axis * ratio**TEMPERATURE_PRESSURE_EXPONENT


def plasma_summary(plasma: dict, source: str = "") -> str:
    """One line naming the plasma and where it came from.

    The log line and the figure caption say the same thing, so they are built
    here rather than twice.  Two separate f-strings drifted apart across a key
    rename and broke the run at the last line of a 60-second job, twice.
    """
    temps = (f"T_i=T_e={plasma['t_hat']:.3g} keV" if "te_hat" not in plasma else
             f"T_i={plasma['ti_hat']:.3g}, T_e={plasma['te_hat']:.3g} keV")  # fmt: skip
    text = (f"n={plasma['n_hat']:.3g}e20 m^-3, {temps}, "
            f"dn/drHat={plasma['dn_drhat']:+.3g}, dT/drHat={plasma['dt_drhat']:+.3g}")  # fmt: skip
    return f"{text} ({source})" if source else text


def _plasma_keys(plasma: dict, collision_operator: str | int = 0) -> dict:
    """Only the keys the namelist template interpolates (``T_i = T_e`` unless split)."""
    return {"n_hat": plasma["n_hat"], "dn_drhat": plasma["dn_drhat"],
            "ti_hat": plasma.get("ti_hat", plasma["t_hat"]),
            "te_hat": plasma.get("te_hat", plasma["t_hat"]),
            "dti_drhat": plasma.get("dti_drhat", plasma["dt_drhat"]),
            "dte_drhat": plasma.get("dte_drhat", plasma["dt_drhat"]),
            "collision_operator": COLLISION_OPERATORS[collision_operator]
            if isinstance(collision_operator, str) else int(collision_operator)}  # fmt: skip


#: ``collision_operator`` names accepted by :func:`run_representative`, and the
#: ``collisionOperator`` each one solves with.  ``pas+momentum_correction``
#: scans ``E_r`` with PAS and corrects the bootstrap current at the evaluated
#: field by the Sugama-Nishimura moment method (:mod:`dkx.momentum_correction`).
COLLISION_OPERATORS = {"fp": 0, "pas": 1, "pas+momentum_correction": 1}


def profile_plasma(profiles: Any, radius: float, a_hat: float) -> dict[str, float]:
    """``&speciesParameters`` values at ``r/a`` from explicit kinetic profiles.

    ``profiles`` carries ``ne_coeffs`` [m^-3], ``Te_coeffs`` and ``Ti_coeffs``
    [eV] as polynomials in ``s``, lowest order first --- attributes (VMEX's
    ``KineticProfiles``) or mapping keys.  This replaces the pressure split of
    :func:`plasma_parameters`, so DKX solves the plasma the equilibrium was
    built with.  Gradients are ``d/drHat = (2 rN / aHat) d/ds``.
    """
    from dkx.bootstrap import _polynomial  # noqa: PLC0415

    def poly(key: str) -> tuple[float, float]:
        coeffs = profiles[key] if isinstance(profiles, Mapping) else getattr(profiles, key)
        return _polynomial(coeffs, float(radius) ** 2)

    (ne, dne), (te, dte), (ti, dti) = poly("ne_coeffs"), poly("Te_coeffs"), poly("Ti_coeffs")
    to_r_hat = 2.0 * float(radius) / float(a_hat)
    return {"n_hat": ne / 1.0e20, "dn_drhat": dne / 1.0e20 * to_r_hat,
            "ti_hat": ti / 1.0e3, "dti_drhat": dti / 1.0e3 * to_r_hat,
            "te_hat": te / 1.0e3, "dte_drhat": dte / 1.0e3 * to_r_hat,
            "t_hat": ti / 1.0e3, "dt_drhat": dti / 1.0e3 * to_r_hat,
            "radius": float(radius)}  # fmt: skip


#: Smallest on-axis density (in 1e20 m^-3) the pressure split may produce.
#: VMEC writes ``presf`` of order 1e-6 Pa for a *vacuum* run rather than exactly
#: zero, and the split turns that into n ~ 1e9 m^-3: a collisionless deck that
#: grinds for tens of minutes and reports transport for a plasma that is not
#: there.  1e-3 is 1e17 m^-3, the same floor ``vmex`` clamps its Redl profiles
#: to, and eleven orders of magnitude below any fusion plasma.
VACUUM_DENSITY_FLOOR = 1.0e-3


#: On-axis pressure of the equilibrium most recently read, so the provenance
#: line can name the (n, T) without a second file open.
_P_AXIS_SEEN: dict[str, float] = {}


def plasma_parameters(equilibrium: Path, radius: float = 0.5) -> dict[str, float]:
    r"""Density and temperature at ``r/a`` from the equilibrium's own pressure.

    The wout carries ``presf`` in Pascals; a drift-kinetic run needs ``n`` and
    ``T`` separately, and pressure alone does not determine them.  One equation,
    two unknowns, so the closure is stated rather than hidden: quasineutral
    hydrogen with ``T_i = T_e = T`` and the pressure split by a power law,

    .. math:: T(s) = T_0 (p(s)/p(0))^{e},
              \qquad n(s) = p(s) / (2 T(s)) \propto p^{1-e},

    with ``T_0`` from :func:`axis_temperature_kev` and ``e =``
    :data:`TEMPERATURE_PRESSURE_EXPONENT`.  Only ``T_0`` is free, and the figure
    states it.

    The temperature must carry a gradient.  A *constant* ``T`` is the simpler
    closure and it is wrong for this figure: the bootstrap current is driven
    mostly by ``dT/dr``, so ``dT/ds = 0`` reports a device with almost no
    bootstrap current and puts the kinetic curve an order of magnitude under
    the equilibrium's own.

    Gradients are returned as ``d/drHat``, matching the deck's
    ``inputRadialCoordinateForGradients = 4`` (the only code that drives the
    potential with ``Er``, which the ambipolar scan needs).  ``rHat = aHat rN``
    with ``aHat = Aminor_p``, so the chain rule ``d/drHat = (1/aHat) d/drN`` is
    applied here rather than left to the caller: omitting it understates the
    drive by a factor of ``aHat``, which is 0.17 on a compact device.

    Returns ``{}`` when the file carries no usable pressure --- absent, or below
    :data:`VACUUM_DENSITY_FLOOR` --- in which case the caller keeps the generic
    reference profile and says so.  Reporting numbers derived from an
    equilibrium the user supplied, without saying how, would be worse than
    reporting generic ones.
    """
    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(equilibrium)) as handle:
            if "presf" not in handle.variables:
                return {}
            pres = np.asarray(handle.variables["presf"][:], dtype=float)
            a_hat = (float(np.asarray(handle.variables["Aminor_p"][...]).reshape(()))
                     if "Aminor_p" in handle.variables else 0.0)  # fmt: skip
    except Exception:
        return {}
    if pres.size < 2 or not np.any(pres > 0.0) or not a_hat > 0.0:
        return {}
    s_grid = np.linspace(0.0, 1.0, pres.size)
    p_axis = float(pres[0])
    # Carried out so the caller can report which (n, T) this run assumed
    # without opening the equilibrium a second time.
    _P_AXIS_SEEN["pa"] = p_axis

    def profiles(s: float) -> tuple[float, float]:
        """``(n [1e20 m^-3], T [keV])`` implied by ``p(s)`` under the closure."""
        p_pa = float(np.interp(s, s_grid, pres))
        t_kev = _temperature(p_pa, p_axis)
        if t_kev <= 0.0:
            return 0.0, 0.0
        return p_pa / (2.0 * t_kev * 1.0e3 * 1.602176634e-19) / 1.0e20, t_kev

    s0 = float(np.clip(radius, 0.0, 1.0)) ** 2  # r/a = sqrt(s)
    n_20, t_kev = profiles(s0)
    if not np.isfinite(n_20) or n_20 < VACUUM_DENSITY_FLOOR:
        return {}
    # Difference in rN and divide by aHat, rather than differencing in s: the
    # deck's gradients are d/drHat and rHat = aHat * rN.
    eps = 1.0e-3
    r_lo, r_hi = max(radius - eps, 0.0), min(radius + eps, 1.0)
    n_lo, t_lo = profiles(r_lo**2)
    n_hi, t_hi = profiles(r_hi**2)
    scale = 1.0 / (a_hat * (r_hi - r_lo))
    return {
        "n_hat": n_20, "t_hat": t_kev,
        "dn_drhat": (n_hi - n_lo) * scale, "dt_drhat": (t_hi - t_lo) * scale,
        "p_pa": float(np.interp(s0, s_grid, pres)), "radius": float(radius),
        "a_hat": a_hat,
    }  # fmt: skip


def _netcdf_text(value: Any) -> str:
    """Decode the fixed-width character arrays used by VMEC metadata."""
    array = np.asarray(value)
    if array.dtype.kind == "S":
        return b"".join(array.reshape((-1,)).tolist()).decode(errors="replace").strip()
    if array.dtype.kind == "U":
        return "".join(array.reshape((-1,)).tolist()).strip()
    return str(array.reshape(()).item()).strip()


def vmec_profile_status(equilibrium: Path) -> dict[str, str]:
    """Classify VMEC profile input without re-evaluating its parameterization.

    VMEC writes canonical evaluated pressure on ``presf`` regardless of whether
    the input used ``power_series``, ``sum_atan``, a spline, or another
    ``pmass_type``. DKX consumes that evaluated profile and retains the input
    representation only as provenance.
    """
    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(equilibrium)) as handle:
            representation = (
                _netcdf_text(handle.variables["pmass_type"][:])
                if "pmass_type" in handle.variables
                else "not_recorded"
            )
            if "presf" not in handle.variables:
                return {
                    "status": "physics_unavailable",
                    "pressure_representation": representation,
                    "detail": "the equilibrium carries no evaluated presf profile",
                }
            pressure = np.asarray(handle.variables["presf"][:], dtype=float)
            if (
                pressure.ndim != 1
                or pressure.size < 2
                or not np.all(np.isfinite(pressure))
            ):
                return {
                    "status": "parser_failure",
                    "pressure_representation": representation,
                    "detail": "presf is not a finite one-dimensional radial profile",
                }
            if "Aminor_p" not in handle.variables:
                return {
                    "status": "physics_unavailable",
                    "pressure_representation": representation,
                    "detail": "the equilibrium carries no minor radius for physical gradients",
                }
    except Exception as exc:
        return {
            "status": "parser_failure",
            "pressure_representation": "unreadable",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    if not plasma_parameters(equilibrium):
        return {
            "status": "physics_unavailable",
            "pressure_representation": representation,
            "detail": _no_pressure_reason(equilibrium),
        }
    return {
        "status": "available",
        "pressure_representation": representation,
        "detail": "using VMEC-evaluated presf with an explicit n/T closure",
    }


def equilibrium_scalars(equilibrium: Path) -> dict[str, Any]:
    """``psiAHat``, ``aHat`` and the VMEC ``<J.B>`` profile, read from the wout.

    ``psiAHat = phi(ns)/(2 pi)`` and ``aHat = Aminor_p`` are how
    ``geometry.F90:130`` and :func:`dkx.magnetic_geometry.psi_a_hat_from_wout`
    define the two flux-surface constants for ``geometryScheme=5``; they set the
    ``psiHat -> rHat`` factor every dimensional flux needs.  ``jdotb`` is VMEC's
    own equilibrium parallel current, in A T/m^2 --- the same quantity DKX
    computes kinetically, so the two can be drawn on one axis.

    Returns ``{}`` if the file cannot be read; every caller treats the
    dimensional overlay as optional.
    """
    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(equilibrium)) as handle:
            out: dict[str, Any] = {}
            if "phi" in handle.variables:
                phi = np.asarray(handle.variables["phi"][:], dtype=float)
                out["psi_a_hat"] = float(phi[-1]) / (2.0 * np.pi)
            if "Aminor_p" in handle.variables:
                out["a_hat"] = float(
                    np.asarray(handle.variables["Aminor_p"][...]).reshape(())
                )
            if "jdotb" in handle.variables:
                jdotb = np.asarray(handle.variables["jdotb"][:], dtype=float)
                out["jdotb"] = jdotb
                out["jdotb_s"] = np.linspace(0.0, 1.0, jdotb.size)
            return out
    except Exception:
        return {}


def resolve_plasma(equilibrium: Path) -> tuple[dict[str, float], str]:
    """The plasma to run, and a phrase saying where it came from.

    Three cases, and the caller must be able to tell them apart on the figure:
    the equilibrium carries a real pressure profile; it carries none; or it
    carries a *vacuum* one, which is the trap.  A vacuum wout's ``presf`` is
    ~1e-6 Pa rather than zero, so the split silently yields n ~ 1e9 m^-3 and a
    collisionless deck --- a 23-minute run on W7-X whose every transport number
    describes a plasma that is not there.
    """
    plasma = plasma_parameters(equilibrium)
    if plasma:
        # Name the assumption. "p(s) from the equilibrium" alone reads as though
        # the whole plasma came from the file; only the pressure did. The
        # bootstrap current moves by a factor of 5 across a plausible range of
        # this number, so a reader comparing against an optimizer's own
        # bootstrap current has to know which temperature was assumed.
        p_axis = float(_P_AXIS_SEEN.get("pa", 0.0))
        n0, t0 = _n_axis_m3(p_axis), _t_axis_kev(p_axis)
        how = (
            "pinned" if _DENSITY_OVERRIDE_M3 is not None
            else "scaled from p(0) against the arXiv:2205.02914 reactor profiles"
        )
        return plasma, (
            f"p(s) from the equilibrium; n(0)={n0:.3g} m^-3 {how}, "
            f"T(0)={t0:.3g} keV from p=2nT (--density-m3 to pin n)"
        )
    return dict(
        FALLBACK_PLASMA
    ), f"generic reference; {_no_pressure_reason(equilibrium)}"


def _no_pressure_reason(equilibrium: Path) -> str:
    """Why the equilibrium's own pressure was not usable, in the reader's terms."""
    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(equilibrium)) as handle:
            if "presf" not in handle.variables:
                return "the equilibrium carries no pressure profile"
            if "Aminor_p" not in handle.variables:
                # Without aHat the d/drHat gradients cannot be formed at all,
                # and assuming 1.0 would understate the drive silently.
                return "the equilibrium carries no minor radius"
            peak = float(np.max(np.asarray(handle.variables["presf"][:], dtype=float)))
    except Exception:
        return "the equilibrium's pressure could not be read"
    return (f"the equilibrium is a vacuum field, p(0)={peak:.3g} Pa"
            if peak < 1.0 else f"p(0)={peak:.3g} Pa is below the plasma floor")  # fmt: skip


def _field_periods(equilibrium: Path) -> int:
    """Field-period count, read from the wout itself.

    For ``geometryScheme=5`` the namelist's ``NPeriods`` stays at its default of
    0 and the real value comes from the equilibrium, so neither the input object
    nor the operator carries it.  Falling back to 1 draws zeta over a full turn
    on an nfp-period device --- an axis wrong by exactly that factor.
    """
    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(equilibrium)) as handle:
            return max(1, int(handle.variables["nfp"][...]))
    except Exception:
        return 1


def _profile_data(equilibrium: Path, *, full: bool = False, quick: bool = False,
                  emit: Callable[[str], None] | None = None,
                  plasma: tuple[dict, str] | None = None,
                  collision_operator: str = "fp") -> dict:  # fmt: skip
    """Geometry plus one RHSMode=1 solve, for the panels the scan cannot fill.

    Returns ``{}`` rather than raising if the profile solve does not apply to
    this equilibrium: a missing panel that says so is better than no figure.
    """
    from dkx.inputs import (  # noqa: PLC0415
        parse_sfincs_input_text,
        sfincs_input_from_raw,
    )
    from dkx.run import run_profile  # noqa: PLC0415

    resolution = _profile_resolution(full=full, quick=quick)
    plasma, derived = plasma or resolve_plasma(equilibrium)
    if emit:
        emit(f"    plasma: {plasma_summary(plasma, derived)}")
        emit(f"    profile resolution: {resolution}")
    text = _PROFILE_TEMPLATE.format(
        equilibrium=str(equilibrium), **resolution, **_plasma_keys(plasma, collision_operator)
    )
    try:
        # run_profile takes a path or an SfincsInput; parse_sfincs_input_text
        # returns a RawNamelist, which it silently mistakes for a path.
        inp = sfincs_input_from_raw(parse_sfincs_input_text(text))
        run = _quiet(lambda: run_profile(inp, out_path=None, emit=None))
    except Exception as exc:  # pragma: no cover - geometry-dependent
        # An out-of-memory here costs three panels, so retry once at half the
        # angular resolution rather than giving up: |B| and a bootstrap number
        # at reduced resolution beat three boxes saying "not present".  The
        # reduced resolution is reported, because a panel at a resolution the
        # caller did not choose must say so.
        # min(v, ...) so a floor can never make an axis GROW: the default n_x is
        # 4, and a bare max(., 5) turned the retry grid into a bigger one.
        reduced = {k: min(v, max(int(v * 2 / 3), 3 if k == "n_x" else 9))
                   for k, v in resolution.items()}  # fmt: skip
        if emit:
            emit(f"    profile solve failed ({type(exc).__name__}); "
                 f"retrying at {reduced}")  # fmt: skip
        try:
            inp = sfincs_input_from_raw(parse_sfincs_input_text(
                _PROFILE_TEMPLATE.format(equilibrium=str(equilibrium), **reduced,
                                         **_plasma_keys(plasma, collision_operator))
            ))  # fmt: skip
            run = _quiet(lambda: run_profile(inp, out_path=None, emit=None))
        except Exception as exc2:
            if emit:
                emit(f"    still unavailable ({type(exc2).__name__}); "
                     f"|B|/bootstrap/flux panels will say so")  # fmt: skip
            return {}
    op, mom = run.operator, run.moments
    # The operator carries no angle grids, and b_hat here is (n_theta, n_zeta)
    # -- the OPPOSITE order from the output-file layout the other panel path
    # sees.  Build the uniform grids explicitly rather than falling back to
    # arange, which silently labels a publication figure in index units.
    nfp = _field_periods(equilibrium)
    data: dict[str, Any] = {
        "resolution": resolution,
        "BHat": np.asarray(op.b_hat),
        "theta": np.linspace(0.0, 2.0 * np.pi, op.n_theta, endpoint=False),
        "zeta": np.linspace(0.0, 2.0 * np.pi / nfp, op.n_zeta, endpoint=False),
    }
    for key in ("FSABjHatOverRootFSAB2", "FSABjHat", "FSABFlow",
                "particleFlux_vm_psiHat", "heatFlux_vm_psiHat"):  # fmt: skip
        if key in mom:
            data[key] = np.asarray(mom[key])
    if emit:
        boot = data.get("FSABjHatOverRootFSAB2", data.get("FSABjHat"))
        if boot is not None:
            emit(
                f"    bootstrap <j.B>/sqrt(<B^2>) = {np.asarray(boot).ravel()[-1]:+.4e}"
            )
    return data


# ---------------------------------------------------------------------------
# Panels that need more than one solve (opt-in via --full)
# ---------------------------------------------------------------------------
#: E_r values (kV/m) for the ambipolarity panel.  Spans the ion root and, on
#: electron-root devices, the positive branch; a root-find would be cheaper but
#: a scan is what shows a reader whether the root is unique.
DEFAULT_ER_SCAN = (-8.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0)




def _ambipolar_roots(records: list[dict[str, Any]]) -> list[float]:
    """Sign changes of ``J_r(E_r)``, linearly interpolated.

    Reported as *bracketed* roots rather than solved ones: the panel's job is to
    show how many there are (one ion root, or the ion/unstable/electron
    triplet), which a Brent solve started from a single guess would hide.

    The interpolation is linear on purpose, and a smarter one is a regression.
    ``J_r = sum_s Z_s Gamma_s`` is a *linear* combination of the moments, and
    linear interpolation is the only scheme that commutes with it: interpolate
    every moment linearly onto the linearly-interpolated root and
    ``sum_s Z_s Gamma_s = 0`` holds exactly there, which is why a Z=+-1 pair
    reports two identical particle fluxes.  A monotone cubic moves the root by
    under 0.01 kV/m on the decks tried and breaks that identity, so the panel
    starts drawing two nearly-coincident Gamma curves that differ only by
    interpolation error.
    """
    roots: list[float] = []
    for a, b in zip(records, records[1:]):
        ja, jb = a["J_r"], b["J_r"]
        if np.isfinite(ja) and np.isfinite(jb) and ja * jb < 0.0:
            roots.append(a["er"] + (b["er"] - a["er"]) * ja / (ja - jb))
    return roots


def _root_types(er: Sequence[float], j_r: Sequence[float],
                radial_factor: float = 1.0) -> list[str]:  # fmt: skip
    """ion / unstable / electron for each root of :func:`_ambipolar_roots`.

    The bracket's secant slope, converted to outward current by
    ``radial_factor`` (the sign of ``drHat/dpsiHat``), feeds the same rule as
    :func:`dkx.er._classify`: a positive outward-current slope is restoring.
    """
    from dkx.er import _classify  # noqa: PLC0415

    types = []
    for ea, eb, ja, jb in zip(er, er[1:], j_r, j_r[1:]):
        if np.isfinite(ja) and np.isfinite(jb) and ja * jb < 0.0:
            root = ea + (eb - ea) * ja / (ja - jb)
            types.append(_classify(float(root), float((jb - ja) / (eb - ea)),
                                   radial_factor=radial_factor))  # fmt: skip
    return types


def _momentum_corrected_fsab_j(deck_text: str, er: float) -> float:
    """``FSABjHat`` of the PAS deck at ``er``, momentum-corrected against its FP twin."""
    from dkx.api import momentum_corrected_bootstrap  # noqa: PLC0415
    from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: PLC0415
    from dkx.namelist import parse_sfincs_input_text  # noqa: PLC0415

    at_er = deck_text.replace("Er = 0.0d+0", f"Er = {float(er)!r}")
    pas, fp = (kinetic_operator_from_namelist(parse_sfincs_input_text(
        at_er.replace("collisionOperator = 1", f"collisionOperator = {c}"))) for c in (1, 0))  # fmt: skip
    return float(np.asarray(_quiet(lambda: momentum_corrected_bootstrap(pas, fp)).fsab_j).ravel()[0])



DEFAULT_SURFACES = (0.25, 0.4, 0.55, 0.7, 0.85)

#: ``E_r`` values (kV/m) bracketing the ion root at every surface.
#:
#: Bounded below at -12 kV/m on purpose.  A Fokker-Planck solve at -25 kV/m on
#: the precise-QA reference does not converge and returns ``<j.B>`` of +30.7
#: against -0.13 everywhere else -- at three different grids, with three
#: different values, so it is a failed solve rather than physics.  Left in the
#: bracket it manufactures a sign change, and :func:`_ambipolar_roots` reports
#: the most negative crossing, so that fake root would win.
DEFAULT_ER_BRACKET = (-12.0, -8.0, -6.0, -4.0, -2.0, -1.0, -0.4, 0.4, 1.0, 2.0)

#: The temperature this bracket was tuned at. The ambipolar field scales with
#: it -- roughly ``E_r ~ T/(e L)`` -- so a bracket sized for a 2 keV plasma sits
#: entirely inside the ion root of a 9 keV one, which is how every surface came
#: to report "no bracketed root" and quote a bootstrap current at the edge.
ER_BRACKET_REFERENCE_T_KEV = 2.0


def _scaled_er_bracket(t_axis_kev: float) -> np.ndarray:
    """The bracket, scaled to the plasma actually being solved."""
    base = np.asarray(DEFAULT_ER_BRACKET, dtype=float)
    if not t_axis_kev > 0.0:
        return base
    return base * (float(t_axis_kev) / ER_BRACKET_REFERENCE_T_KEV)


#: How far past the bracket's negative edge the scan may reach when the
#: evidence says a root is just outside, and the step it takes getting there.
#: Bounded rather than open-ended: the point is to catch a root a volt or two
#: beyond the edge, not to go hunting at fields no device sustains.
#: The ion root deepens roughly with temperature, so a reactor-scale plasma at
#: 9 keV sits far outside a bracket sized for a 2 keV one. The floor is
#: therefore generous and the *step* is what keeps the cost down: each probe
#: extrapolates to where J_r is heading rather than walking out in fixed jumps.
ER_EXTENSION_FLOOR_KV_M = -120.0
#: Cap on probes per surface, so a scan that never crosses cannot run away.
ER_EXTENSION_MAX_PROBES = 6
#: Smallest step, so extrapolation from a nearly flat pair still makes progress.
ER_EXTENSION_MIN_STEP_KV_M = 2.0


#: Root refinement after bracketing: SciPy's Brent method on the bracket, at
#: most this many extra solves, stopping once |J_r| <= ER_ROOT_RTOL max|J_r|
#: (far above the 1e-10 kinetic residual, so solver noise cannot stall it) or
#: the bracket is narrower than ER_ROOT_XTOL_KV_M.
ER_ROOT_MAX_SOLVES = 4
ER_ROOT_RTOL = 1.0e-6
ER_ROOT_XTOL_KV_M = 0.01


def _refine_root(solve: Callable[[float], tuple[float, Any]], er: np.ndarray,
                 j_r: np.ndarray, root: float) -> tuple[float, list[tuple[float, float, Any]]]:
    """Refine a bracketed linear-interpolation root of ``J_r(E_r)``.

    A 10-20 kV/m bracket interpolated linearly misplaces a curved ``J_r`` by
    several kV/m, enough to make a radial ``E_r`` profile non-monotone.
    ``solve(er)`` returns ``(J_r, scan)`` for one field; every evaluation is
    returned so the caller reads moments *at* the root, not interpolated.
    """
    from scipy.optimize import brentq  # noqa: PLC0415

    known = dict(zip(np.asarray(er, float).tolist(), np.asarray(j_r, float).tolist()))
    e = np.array(sorted(known))
    i = int(np.clip(np.searchsorted(e, root) - 1, 0, e.size - 2))
    ftol = ER_ROOT_RTOL * float(np.nanmax(np.abs(list(known.values()))))
    evals: list[tuple[float, float, Any]] = []

    class _Done(Exception):
        pass

    def f(x: float) -> float:
        if x in known:  # brentq re-reads the bracket ends: already solved
            return known[x]
        if len(evals) == ER_ROOT_MAX_SOLVES:
            raise _Done
        fx, scan = solve(x)
        evals.append((x, fx, scan))
        if not np.isfinite(fx) or abs(fx) <= ftol:
            raise _Done
        return fx

    try:
        x = brentq(f, e[i], e[i + 1], xtol=ER_ROOT_XTOL_KV_M, maxiter=ER_ROOT_MAX_SOLVES + 1, disp=False)
    except _Done:
        x = root
    finite = [(abs(fx), x_) for x_, fx, _ in evals if np.isfinite(fx)]
    return (min(finite)[1] if finite else x), evals


def _next_er_probe(er: np.ndarray, j_r: np.ndarray) -> float:
    """Where to sample next: the linear extrapolation to ``J_r = 0``.

    Walking out in fixed steps costs one solve per step and has no idea how far
    it has to go -- at 2 keV the root sits a couple of volts past the edge and
    at 9 keV it is tens. Extrapolating from the two most negative samples aims
    at the crossing directly, and because the predicate only extends while
    ``J_r`` is approaching zero the extrapolation always points outward.

    Overshoot is deliberate but bounded: the estimate is stepped past the
    predicted crossing by a quarter so the new point is likely to *straddle*
    it rather than land just short and need another probe.
    """
    order = np.argsort(er)
    e, j = np.asarray(er)[order], np.asarray(j_r)[order]
    finite = np.isfinite(j)
    e, j = e[finite], j[finite]
    slope = (j[1] - j[0]) / (e[1] - e[0])
    if not np.isfinite(slope) or slope == 0.0:
        return float(e[0]) - ER_EXTENSION_MIN_STEP_KV_M
    crossing = float(e[0] - j[0] / slope)
    step = max(ER_EXTENSION_MIN_STEP_KV_M, 1.25 * (float(e[0]) - crossing))
    return float(e[0]) - step


def _wants_more_negative_er(er: np.ndarray, j_r: np.ndarray) -> bool:
    """Whether ``J_r`` says a root sits just past the negative edge.

    Three conditions, all necessary. There must be no sign change already
    (otherwise a root is in hand); ``J_r`` at the most negative sampled field
    must be finite and still on one side of zero; and it must be *heading*
    toward zero as the field decreases, i.e. smaller in magnitude at the edge
    than at its neighbour. A scan whose current grows toward the edge has no
    root out there and extending would only cost solves.

    This is deliberately not "extend whenever no root was found". The bracket's
    negative floor exists because, on at least one reference equilibrium, a
    failed solve at a more negative field manufactured a sign change and the
    root finder reported the fake crossing. Extending only into a monotone
    approach keeps that case out: a failed solve does not approach zero
    monotonically.
    """
    order = np.argsort(er)
    e, j = np.asarray(er)[order], np.asarray(j_r)[order]
    finite = np.isfinite(j)
    if finite.sum() < 2:
        return False
    e, j = e[finite], j[finite]
    if np.any(np.sign(j[:-1]) * np.sign(j[1:]) <= 0.0):
        return False  # a crossing is already bracketed
    return bool(abs(j[0]) < abs(j[1]))


def _interp_at_root(er: np.ndarray, values: np.ndarray, root: float) -> float:
    """Linear interpolation of a scanned quantity onto the ambipolar root."""
    order = np.argsort(er)
    return float(np.interp(root, np.asarray(er)[order], np.asarray(values)[order]))


def _root_fsab2(moments: Mapping[str, Any] | dict[str, Any]) -> float | None:
    """``sqrt(<B^2>)/BBar`` for this surface, from two moments the scan already has.

    ``FSABjHat / FSABjHatOverRootFSAB2`` is exactly ``sqrt(FSABHat2)`` by the
    definitions in documentation eq. (194) and (196), and it is a property of
    the geometry alone, so it is the same at every scanned ``E_r``.  Taking the
    least-squares slope over the whole scan rather than one ratio keeps the
    estimate finite when an individual current passes through zero.
    """
    num = moments.get("FSABjHat")
    den = moments.get("FSABjHatOverRootFSAB2")
    if num is None or den is None:
        return None
    num = np.asarray(num, dtype=float).ravel()
    den = np.asarray(den, dtype=float).ravel()
    denominator = float(np.dot(den, den))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return None
    slope = float(np.dot(num, den)) / denominator
    return slope if np.isfinite(slope) and slope > 0.0 else None


def _vmec_current_kA_m2(geometry: dict[str, Any], radius: float,
                        root_fsab2: float) -> float | None:  # fmt: skip
    """VMEC's own ``<J.B>`` at ``r/a = radius``, in the DKX panel's units.

    The wout ``jdotb`` is ``<J.B>`` in A T/m^2 on the full mesh in ``s``; the
    DKX curve is ``<j.B>/sqrt(<B^2>)`` in kA/m^2, so dividing by the same
    ``sqrt(<B^2>)`` puts the equilibrium and the kinetic current on one axis.
    """
    if "jdotb" not in geometry or root_fsab2 <= 0.0:
        return None
    jdotb = np.asarray(geometry["jdotb"], dtype=float)
    s_grid = np.asarray(geometry["jdotb_s"], dtype=float)
    if jdotb.size < 2:
        return None
    return float(np.interp(radius**2, s_grid, jdotb)) / root_fsab2 / 1.0e3


def radial_profiles(
    equilibrium: Path,
    *,
    surfaces: Sequence[float] | None = None,
    er_values: Sequence[float] | None = None,
    full: bool = False,
    quick: bool = False,
    emit: Callable[[str], None] | None = None,
    profiles: Any = None,
    er: float | None = None,
    collision_operator: str = "fp",
) -> list[dict[str, Any]]:
    """Ambipolar ``E_r`` and evaluated moments, per surface.

    Prefer the most negative *stable* bracketed root (stability from the
    outward-current slope, :func:`dkx.er._classify`); otherwise retain the
    sampled point with smallest ``|J_r|`` and mark ``evaluation_is_root=False``.
    ``er`` (kV/m) prescribes the field instead of searching for a root.

    Only fields whose solve passes the original-equation check
    (``algebraic_converged``) are bracketed; the others are listed in
    ``uncertified_er``. Each surface's scan also samples the neighbouring surface's
    root (``er_seed``), and a failure on one surface is recorded as
    ``evaluation_status="solve_failure"`` without stopping the others. Each
    record carries the ``resolution`` it was solved at; that is not a
    resolution-convergence certificate.  Each
    surface is solved with its *own* ``n``, ``T`` and gradients: from
    ``profiles`` (:func:`profile_plasma`) when given, else the pressure split
    evaluated at that radius.
    """
    import tempfile  # noqa: PLC0415

    from dkx.api import batched_er_scan  # noqa: PLC0415
    from dkx.units import (  # noqa: PLC0415
        CURRENT_DENSITY,
        HEAT_FLUX,
        PARTICLE_FLUX,
        flux_psi_hat_to_r_hat,  # noqa: PLC0415
    )

    resolution = _profile_resolution(full=full, quick=quick)
    # Defaulted through ``None`` rather than in the signature: ``quick`` has to
    # be able to pick the smaller surface list and bracket, and a signature
    # default would already have chosen for it.
    if surfaces is None:
        surfaces = QUICK_SURFACES if quick else DEFAULT_SURFACES
    if collision_operator not in COLLISION_OPERATORS:
        raise ValueError(f"collision_operator must be one of {sorted(COLLISION_OPERATORS)}")
    plasma, _derived = resolve_plasma(equilibrium)
    geometry = equilibrium_scalars(equilibrium)
    prescribed_er = er
    if er is not None:
        er_values = [float(er)]
    if er_values is None:
        base = QUICK_ER_BRACKET if quick else DEFAULT_ER_BRACKET
        # Scale to this plasma, not to the 2 keV one the tuple was tuned on.
        t_axis = (profile_plasma(profiles, 0.0, 1.0)["te_hat"] if profiles is not None
                  else _t_axis_kev(_P_AXIS_SEEN.get("pa", 0.0)))  # fmt: skip
        er_values = _scaled_er_bracket(t_axis)
        if len(base) != len(DEFAULT_ER_BRACKET):
            er_values = np.asarray(base, dtype=float) * (
                float(er_values[0]) / float(DEFAULT_ER_BRACKET[0])
            )
    er = np.asarray(er_values, dtype=float)
    if emit:
        emit(f"    radial-scan resolution: {resolution}; "
             f"{len(er)} Er points per surface over "
             f"[{er.min():.3g}, {er.max():.3g}] kV/m")  # fmt: skip
    profile_input = vmec_profile_status(equilibrium)
    if profiles is not None:
        profile_input = {"status": "available", "pressure_representation": "explicit",
                         "detail": "explicit n, T profiles (polynomials in s)"}  # fmt: skip
    # Outward-current sign of the psiHat current: psiHat = psiAHat rN^2.
    radial_factor = -1.0 if geometry.get("psi_a_hat", 1.0) < 0.0 else 1.0

    def certified(deck: Path, fields: Any) -> tuple:
        """One batched scan, keeping only fields that passed the original-equation check.

        Returns ``(er, J_r, moments, uncertified_er)``. An uncertified field is
        reported, never bracketed: a failed solve can manufacture a sign change.
        """
        fields = np.asarray(fields, dtype=float).ravel()
        sc = _quiet(lambda: batched_er_scan(deck, fields, retain_full_state=True))
        j = np.asarray(sc.radial_current, dtype=float).ravel()
        ok = getattr(sc, "algebraic_converged", None)
        keep = np.isfinite(j) & (True if ok is None else np.asarray(ok, dtype=bool).ravel())
        moments = {k: np.asarray(v)[keep] if np.ndim(v) and len(v) == len(fields) else v
                   for k, v in sc.moments.items()}  # fmt: skip
        return fields[keep], j[keep], moments, fields[~keep].tolist()

    def merge(a: tuple, b: tuple) -> tuple:
        """Two certified scans as one, sorted by field."""
        er_ab = np.concatenate([a[0], b[0]])
        order = np.argsort(er_ab, kind="stable")
        per_field = lambda m, n: {k: np.asarray(v) for k, v in m.items() if np.ndim(v) and len(v) == n}  # noqa: E731
        ma, mb = per_field(a[2], len(a[0])), per_field(b[2], len(b[0]))
        moments = {**a[2], **{k: np.concatenate([ma[k], mb[k]])[order] for k in ma if k in mb}}
        return er_ab[order], np.concatenate([a[1], b[1]])[order], moments, a[3] + b[3]

    out: list[dict[str, Any]] = []
    seed: float | None = None  # radial continuation: the neighbouring surface's root
    with tempfile.TemporaryDirectory() as work:
        for radius in surfaces:
            local = (profile_plasma(profiles, radius, geometry["a_hat"]) if profiles is not None
                     else plasma_parameters(equilibrium, radius) or plasma)  # fmt: skip
            text = _PROFILE_TEMPLATE.format(
                equilibrium=str(equilibrium), **resolution,
                **_plasma_keys(local, collision_operator),
            ).replace("rN_wish = 0.5", f"rN_wish = {radius}")  # fmt: skip
            deck = Path(work) / f"in_{radius}.namelist"
            deck.write_text(text)
            base = {"r": float(radius), "profile_input_status": profile_input["status"],
                    "pressure_representation": profile_input["pressure_representation"],
                    "profile_input_detail": profile_input["detail"],
                    "resolution": dict(resolution), "er_seed": seed}  # fmt: skip
            fields = er if prescribed_er is not None or seed is None else np.unique(np.r_[er, seed])
            try:
                scan = certified(deck, fields)
                er_used, j_r, moments, uncertified = scan
                roots = _ambipolar_roots([{"er": float(e), "J_r": float(j)}
                                          for e, j in zip(er_used, j_r)])  # fmt: skip
                # A root just past the negative edge is common on a device whose
                # ion root deepens with radius; extend while the evidence says a
                # root is out there and stop as soon as one is bracketed. Probe
                # moments merge into the scan, so no field is solved twice.
                probes: list[float] = []
                while (
                    prescribed_er is None
                    and not roots
                    and len(probes) < ER_EXTENSION_MAX_PROBES
                    and _wants_more_negative_er(er_used, j_r)
                ):
                    edge = _next_er_probe(er_used, j_r)
                    if edge < ER_EXTENSION_FLOOR_KV_M:
                        break
                    probes.append(edge)
                    probe = certified(deck, [edge])
                    er_used, j_r, moments, uncertified = merge((er_used, j_r, moments, uncertified), probe)
                    if probe[3]:
                        break  # an uncertified edge ends the extension; it is reported
                    roots = _ambipolar_roots([{"er": float(e), "J_r": float(j)}
                                              for e, j in zip(er_used, j_r)])  # fmt: skip
                er_grid = er_used
                types = _root_types(er_grid, j_r, radial_factor)
                record: dict[str, Any] = {
                    **base, "er_scan": er_grid.tolist(), "J_r": j_r.tolist(),
                    "roots": roots, "root_types": types, "uncertified_er": uncertified,
                }  # fmt: skip
                evaluation_er: float | None = None
                if prescribed_er is not None:
                    record["evaluation_status"] = "prescribed_er"
                    record["evaluation_is_root"] = False
                    record["er_prescribed"] = evaluation_er = float(prescribed_er)
                elif roots:
                    # The ion root is the most negative *stable* crossing; a device
                    # in the electron-root regime has a positive one too, and the
                    # unstable middle root is never a state the plasma occupies.
                    stable = [x for x, t in zip(roots, types) if t in ("ion", "electron")]
                    root = min(stable or roots)

                    def one(e, d=deck):
                        point = certified(d, [e])
                        uncertified.extend(point[3])
                        return (float(point[1][0]) if point[1].size else float("nan")), point

                    bracketed, (root, evals) = root, _refine_root(one, er_grid, j_r, root)
                    record["roots"] = [root if r == bracketed else r for r in roots]
                    for _x, _f, point in evals:
                        er_grid, j_r, moments, _ = merge((er_grid, j_r, moments, []), point)
                    record["er_scan"], record["J_r"] = er_grid.tolist(), j_r.tolist()
                    record["er_root_refine_solves"] = len(evals)
                    record["er_ambipolar"] = seed = root
                    record["evaluation_status"] = "bracketed_root"
                    record["evaluation_is_root"] = True
                    evaluation_er = root
                elif j_r.size:
                    closest = int(np.argmin(np.abs(j_r)))
                    evaluation_er = float(er_grid[closest])
                    record["evaluation_status"] = "no_bracketed_root"
                    record["evaluation_is_root"] = False
                    record["radial_current_evaluated"] = float(j_r[closest])
                else:
                    record["evaluation_status"] = "solve_failure"
                    record["failure_type"] = "NoCertifiedRadialCurrent"
                    record["failure_detail"] = (
                        "no field of the electric-field scan passed the original-equation check"
                    )
            except Exception as exc:  # noqa: BLE001 - recorded per surface, the scan goes on
                if emit:
                    emit(f"    r/a={radius:.2f}: solve failure ({type(exc).__name__})")
                out.append({**base, "evaluation_status": "solve_failure",
                            "failure_type": type(exc).__name__, "failure_detail": str(exc)})  # fmt: skip
                continue
            if evaluation_er is not None:
                record["er_evaluated"] = evaluation_er
                mom = moments
                boot = mom.get("FSABjHatOverRootFSAB2", mom.get("FSABjHat"))
                if boot is not None:
                    value = _interp_at_root(er_grid, np.asarray(boot).ravel(), evaluation_er)
                    record["bootstrap"] = value
                    # <j.B>/sqrt(<B^2>) carries e nBar vBar (documentation
                    # eq. 196), so this is the same current in kA/m^2.
                    record["bootstrap_kA_m2"] = value * CURRENT_DENSITY / 1.0e3
                root_fsab2 = _root_fsab2(mom)
                if root_fsab2 is not None:
                    record["root_fsab2"] = root_fsab2
                    vmec = _vmec_current_kA_m2(geometry, radius, root_fsab2)
                    if vmec is not None:
                        record["jdotb_vmec_kA_m2"] = vmec
                    if collision_operator == "pas+momentum_correction":
                        value = _momentum_corrected_fsab_j(text, evaluation_er) / root_fsab2
                        record["bootstrap"] = value
                        record["bootstrap_kA_m2"] = value * CURRENT_DENSITY / 1.0e3
                # Fluxes are computed in the psiHat coordinate; the radial flux
                # density a reader wants is the rHat one (eq. 175), and the RBar
                # in its normalization then cancels (see dkx.units).
                to_r_hat = None
                if "psi_a_hat" in geometry and "a_hat" in geometry:
                    to_r_hat = flux_psi_hat_to_r_hat(
                        psi_a_hat=geometry["psi_a_hat"],
                        a_hat=geometry["a_hat"],
                        r_n=radius,
                    )
                for key, name, unit in (
                    ("particleFlux_vm_psiHat", "particle_flux", PARTICLE_FLUX),
                    ("heatFlux_vm_psiHat", "heat_flux", HEAT_FLUX / 1.0e3),
                ):  # fmt: skip
                    if key in mom:
                        arr = np.asarray(mom[key])
                        values = [
                            _interp_at_root(er_grid, arr[:, s], evaluation_er)
                            for s in range(arr.shape[1])
                        ]
                        record[name] = values
                        if to_r_hat is not None:
                            record[f"{name}_si"] = [v * to_r_hat * unit for v in values]
            if emit:
                e_txt = f"{record.get('er_evaluated', float('nan')):+.3f}"
                b_txt = f"{record.get('bootstrap_kA_m2', float('nan')):+.4g}"
                status = str(record.get("evaluation_status", "unavailable"))
                line = (f"    r/a={radius:.2f}  Er_evaluated={e_txt} kV/m "
                        f"status={status}  "
                        f"<j.B>/sqrt(<B^2>)={b_txt} kA/m^2")  # fmt: skip
                if not record.get("evaluation_is_root", False):
                    # Without this the number reads as a transport result. It is
                    # not: with no sign change in the sampled interval the scan
                    # falls back to whichever sampled field had the smallest
                    # |J_r|, which is typically an endpoint. A bootstrap current
                    # at a non-ambipolar field is not a state the plasma
                    # occupies, and quoting it beside genuine roots invites
                    # exactly the comparison it cannot support.
                    line += "  [NOT an ambipolar root: J_r != 0 here]"
                if record["uncertified_er"]:
                    line += f"  [{len(record['uncertified_er'])} uncertified fields excluded]"
                emit(line)
            out.append(record)
    return out



def write_representative_output(
    path: str | Path,
    *,
    scan: list[dict[str, Any]] | None = None,
    profiles: list[dict[str, Any]] | None = None,
    data: dict[str, Any] | None = None,
    equilibrium: str | Path | None = None,
    resolution_profile: dict[str, int] | None = None,
    resolution_monoenergetic: dict[str, int] | None = None,
) -> Path:
    """Persist the numbers behind the figure.

    A run that leaves only a PNG cannot be re-plotted, re-scaled, or checked
    against another code, so ``dkx wout_*.nc`` always writes this alongside it.
    HDF5 when ``h5py`` is available, JSON otherwise --- the point is that the
    data survives, not the container.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from dkx import units  # noqa: PLC0415

    payload: dict[str, Any] = {
        "equilibrium": str(equilibrium) if equilibrium else "",
        "resolution_monoenergetic": resolution_monoenergetic or DEFAULT_RESOLUTION,
        "resolution_profile": resolution_profile or DEFAULT_RESOLUTION_PROFILE,
        # The SI factors behind every "_kA_m2" / "_si" dataset, so the file can
        # be converted back to SFINCS units without consulting the source.
        "units": {
            "current_density_A_per_m2": units.CURRENT_DENSITY,
            "particle_flux_per_m2_s": units.PARTICLE_FLUX,
            "heat_flux_W_per_m2": units.HEAT_FLUX,
        },
    }
    if scan:
        for key in ("nu_prime", "e_star", "D11", "D31", "D33"):
            payload[f"monoenergetic/{key}"] = [r[key] for r in scan]
    if profiles:
        payload["profiles/r"] = [p["r"] for p in profiles]
        for key in ("er_ambipolar", "radial_current_evaluated", "bootstrap",
                    "bootstrap_kA_m2", "jdotb_vmec_kA_m2", "jdotb_redl_kA_m2",
                    "root_fsab2", "er_prescribed"):  # fmt: skip
            payload[f"profiles/{key}"] = [p.get(key, float("nan")) for p in profiles]
        # ``er_evaluated`` was added after ``er_ambipolar``.  Preserve older
        # callers that provide a root-only profile: a known root is necessarily
        # the field at which its observables were evaluated.
        payload["profiles/er_evaluated"] = [
            p.get("er_evaluated", p.get("er_ambipolar", float("nan"))) for p in profiles
        ]
        payload["profiles/evaluation_is_root"] = [
            float(bool(p.get("evaluation_is_root", "er_ambipolar" in p)))
            for p in profiles
        ]
        for key in (
            "evaluation_status",
            "profile_input_status",
            "pressure_representation",
            "profile_input_detail",
            "failure_type",
            "failure_detail",
        ):
            payload[f"profiles/{key}"] = [str(p.get(key, "")) for p in profiles]
        for key in ("particle_flux", "heat_flux", "particle_flux_si", "heat_flux_si"):
            rows = [p.get(key) for p in profiles if p.get(key) is not None]
            if rows:
                payload[f"profiles/{key}"] = rows
    if data and "BHat" in data:
        for key in ("BHat", "theta", "zeta"):
            if key in data:
                payload[f"geometry/{key}"] = np.asarray(data[key]).tolist()

    try:
        import h5py  # noqa: PLC0415

        with h5py.File(path, "w") as handle:
            for key, value in payload.items():
                if isinstance(value, str):
                    handle.attrs[key] = value
                elif isinstance(value, dict):
                    for sub, v in value.items():
                        handle.attrs[f"{key}/{sub}"] = v
                else:
                    array = np.asarray(value)
                    if array.dtype.kind in {"U", "S", "O"}:
                        string_type = h5py.string_dtype(encoding="utf-8")
                        handle.create_dataset(
                            key, data=array.astype(object), dtype=string_type
                        )
                    else:
                        handle.create_dataset(key, data=np.asarray(value, dtype=float))
        return path.resolve()
    except Exception:
        import json  # noqa: PLC0415

        out = path.with_suffix(".json")
        out.write_text(json.dumps(payload, indent=1) + "\n")
        return out.resolve()
