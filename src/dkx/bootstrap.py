"""Kinetic bootstrap current as an objective term on a VMEX equilibrium.

``vmex.core.bootstrap.RedlBootstrapMismatch`` drives a VMEC equilibrium toward
the Redl *analytic* bootstrap current.  This module is the drift-kinetic
counterpart: DKX solves the linearized drift-kinetic equation on the same
equilibrium and returns :math:`\\langle j_\\parallel B\\rangle` itself.  Redl
is a fit valid in the banana regime for a quasisymmetric field; the kinetic
current is a solve on the actual geometry.

Two terms, one per derivative lane of ``vmex.optimize``:

- :class:`KineticBootstrapMismatch` is traced (VMEX state -> Boozer spectrum
  -> DKX), so VMEX's implicit Jacobian differentiates it.  Same residual and
  interface as ``RedlBootstrapMismatch``; one import and one tuple::

      from dkx.bootstrap import KineticBootstrapMismatch

      kinetic = KineticBootstrapMismatch(profiles, surfaces=SURFACES)
      objective_function_terms.append((kinetic, 0.0, KINETIC_WEIGHT))

- :class:`KineticBootstrapCurrent` is a host term on the written wout
  (VMEC-file route, optionally at the ambipolar root), for problems built with
  ``derivative_method="finite_difference"``.  Each residual evaluation costs
  one DKX solve per surface (per ``E_r`` point when ``ambipolar=True``), so
  keep ``surfaces`` short.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dkx.units import PARALLEL_CURRENT

__all__ = ["DEFAULT_COLLISION_OPERATOR", "DEFAULT_ER_BRACKET", "DEFAULT_RESOLUTION",
           "KineticBootstrapCurrent", "KineticBootstrapMismatch",
           "plot_kinetic_bootstrap_current"]  # fmt: skip

#: Grid for one bootstrap-current solve.  Nxi >= Nzeta because the Legendre
#: resolution is what limits accuracy at low collisionality; Nx=5 is the
#: smallest speed grid that keeps the current converged for these profiles.
DEFAULT_RESOLUTION: dict[str, int] = {"n_theta": 21, "n_zeta": 31, "n_xi": 32, "n_x": 5}

#: ``E_r`` values (kV/m) bracketing the ion root, used only when the ambipolar
#: root is requested.
DEFAULT_ER_BRACKET: tuple[float, ...] = (-8.0, -4.0, -2.0, -1.0, -0.4, 0.4, 1.0, 2.0)

#: The full linearized Fokker-Planck operator, and not a cheaper one.  The
#: bootstrap current *is* the parallel-momentum moment, which is exactly what
#: pitch-angle scattering gets wrong: it has no momentum-restoring term.
#: Measured against Redl on the finite-beta precise-QA equilibrium
#: ``wout_LandremanPaul2021_QA_beta2p5_bootstrap`` at s = 0.25/0.5/0.75, with
#: profiles fixed by that equilibrium's own p(0):
#:
#: ===========================  ======================  =======
#: ``collisionOperator``        ratio to Redl           cost
#: ===========================  ======================  =======
#: 1 (pitch-angle scattering)   1.47, 1.42, 1.35        29 s
#: 3 (Sugama)                   0.74, 0.77, 0.80        62 s
#: **0 (Fokker-Planck)**        **0.93, 0.95, 0.98**    56 s
#: ===========================  ======================  =======
#:
#: Redl is a fit to Fokker-Planck calculations, so 2-7% is the agreement to
#: expect and 35-47% is a bias, not a physical difference.  Under 2x the cost
#: buys it.
DEFAULT_COLLISION_OPERATOR: int = 0

#: Residual scale in A T/m^2: one MA T/m^2, so a residual row *is* the surface's
#: ``<j.B>`` in the units VMEX plots it in.  Fixed rather than derived from the
#: equilibrium, so the objective landscape does not move under the optimizer;
#: the tuple weight carries the rest of the scaling.
DEFAULT_REFERENCE_CURRENT: float = 1.0e6

_TEMPLATE = """&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 5
  equilibriumFile = "{equilibrium}"
  VMECRadialOption = 0
  inputRadialCoordinate = 3
  rN_wish = {r_n:.10g}
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = {n_hat:.10g} {n_hat:.10g}
  THats = {ti_hat:.10g} {te_hat:.10g}
  dNHatdrHats = {dn_drhat:.10g} {dn_drhat:.10g}
  dTHatdrHats = {dti_drhat:.10g} {dte_drhat:.10g}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 8.330d-3
  Er = {er:.10g}
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


def _minor_radius(equilibrium: str | Path) -> float:
    """``aHat = Aminor_p`` from the wout (``geometry.F90:130``), 1.0 if unreadable."""
    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(equilibrium)) as handle:
            value = float(np.asarray(handle.variables["Aminor_p"][...]).reshape(()))
    except Exception:
        return 1.0
    return value if value > 0.0 else 1.0


def _polynomial(coefficients: Any, s: float) -> tuple[float, float]:
    """Value and ``d/ds`` of ``sum_k c[k] s**k`` (the simsopt/VMEX convention)."""
    c = np.atleast_1d(np.asarray(coefficients, dtype=float))
    value = 0.0
    dds = 0.0
    for k in range(c.size - 1, -1, -1):
        dds = dds * s + value
        value = value * s + float(c[k])
    return value, dds


@dataclass
class KineticBootstrapCurrent:
    """DKX ``<j.B>`` on ``surfaces`` of a VMEX equilibrium, as a residual vector.

    ``profiles`` is any object carrying ``ne_coeffs`` [m^-3],
    ``Te_coeffs`` [eV] and ``Ti_coeffs`` [eV] --- polynomial coefficients in
    ``s``, lowest order first.  ``vmex.core.bootstrap.KineticProfiles`` is
    exactly that, so the same profile object feeds this term and the Redl one
    and the two are guaranteed to describe the same plasma.

    The residual for surface :math:`i` is
    :math:`\\langle j_\\parallel B\\rangle_i / \\mathrm{reference\\_current}`
    against target 0, so with the default scale a residual row is that surface's
    current in MA T/m\\ :sup:`2`.  A finite-beta stellarator carries a few tenths
    of that, which puts the rows in the same range as a Redl ``f_boot`` term and
    makes the tuple weight the only knob worth turning.

    ``er_kV_per_m`` is the prescribed radial electric field; the default of zero
    is the standard choice for a bootstrap-current objective, because
    :math:`\\langle j_\\parallel B\\rangle` depends on :math:`E_r` far more
    weakly than the radial fluxes do.  Set ``ambipolar=True`` to solve for the
    ambipolar root at every surface instead --- physically the better answer,
    and ``len(er_values)`` times the cost.
    """

    profiles: Any
    surfaces: Sequence[float] = (0.25, 0.5, 0.75)
    resolution: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_RESOLUTION))
    er_kV_per_m: float = 0.0
    ambipolar: bool = False
    er_values: Sequence[float] = DEFAULT_ER_BRACKET
    reference_current: float = DEFAULT_REFERENCE_CURRENT
    collision_operator: int = DEFAULT_COLLISION_OPERATOR
    verbose: bool = False

    name = "j_boot_dkx"

    def __post_init__(self) -> None:
        self.surfaces = np.atleast_1d(np.asarray(self.surfaces, dtype=float))
        self._cache: tuple[Any, np.ndarray] | None = None

    # -- plasma and deck ------------------------------------------------------

    def plasma_at(self, s: float, *, a_hat: float) -> dict[str, float]:
        """The ``&speciesParameters`` entries for surface ``s``.

        SFINCS normalizes to ``nBar = 1e20 m^-3`` and ``TBar = 1 keV``
        (:mod:`dkx.units`).  The deck leaves
        ``inputRadialCoordinateForGradients`` at the v3 default of 4, because
        that is the only code that drives the potential with ``Er`` --- any
        other choice raises "Er != 0 with a non-Er
        inputRadialCoordinateForGradients", which ``ambipolar=True`` would hit
        immediately.  Code 4 wants ``d/drHat``, and the profile polynomials are
        in ``s``, so the chain rule is
        ``d/drHat = (1/aHat) d/drN = (2 sqrt(s) / aHat) d/ds``.
        """
        ne, dne = _polynomial(self.profiles.ne_coeffs, s)
        te, dte = _polynomial(self.profiles.Te_coeffs, s)
        ti, dti = _polynomial(self.profiles.Ti_coeffs, s)
        to_r_hat = 2.0 * float(np.sqrt(max(s, 0.0))) / float(a_hat)
        return {
            "n_hat": ne / 1.0e20, "dn_drhat": dne / 1.0e20 * to_r_hat,
            "te_hat": te / 1.0e3, "dte_drhat": dte / 1.0e3 * to_r_hat,
            "ti_hat": ti / 1.0e3, "dti_drhat": dti / 1.0e3 * to_r_hat,
        }  # fmt: skip

    def namelist(self, equilibrium: str | Path, s: float, *, er: float,
                 a_hat: float | None = None) -> str:  # fmt: skip
        """The full DKX input deck for one surface of one equilibrium.

        ``a_hat`` defaults to the equilibrium's own ``Aminor_p``; pass it
        explicitly only to build a deck without touching the file.
        """
        if a_hat is None:
            a_hat = _minor_radius(equilibrium)
        return _TEMPLATE.format(
            equilibrium=str(equilibrium), r_n=float(np.sqrt(max(s, 0.0))), er=float(er),
            collision_operator=int(self.collision_operator),
            **self.plasma_at(float(s), a_hat=a_hat), **self.resolution,
        )  # fmt: skip

    # -- evaluation -----------------------------------------------------------

    def current_profile(self, eq: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(s, <j.B>)`` in A T/m^2 --- the unit of the VMEC ``jdotb``.

        Surfaces DKX cannot solve are reported as ``nan`` rather than as zero:
        a failed solve is not a device with no bootstrap current.
        """
        return np.asarray(self.surfaces, dtype=float), self._evaluate(eq)

    def residuals(self, eq: Any) -> np.ndarray:
        """``<j.B> / reference_current`` per surface; target 0."""
        values = self._evaluate(eq)
        return np.nan_to_num(values, nan=0.0) / float(self.reference_current)

    def profile(self, eq: Any) -> np.ndarray:
        """Per-surface squared residuals (``sum = total``)."""
        r = self.residuals(eq)
        return r * r

    def total(self, eq: Any) -> float:
        """Scalar objective ``sum(residuals**2)``."""
        r = self.residuals(eq)
        return float(np.sum(r * r))

    def J(self, eq: Any) -> np.ndarray:
        """Objective-term entry point for ``vmex.optimize`` least squares."""
        return self.residuals(eq)

    __call__ = J

    # -- internals ------------------------------------------------------------

    def _evaluate(self, eq: Any) -> np.ndarray:
        """One DKX solve per surface, memoized for the equilibrium object itself.

        The reporter asks for ``total`` right after the optimizer has asked for
        ``residuals`` on the same equilibrium; without this the second call pays
        for every solve again.  The cache holds one entry and keeps a strong
        reference to the equilibrium, because identity is the key and a freed
        object's ``id`` can be handed to its successor.
        """
        cached = self._cache
        if cached is not None and cached[0] is eq:
            return cached[1]
        with tempfile.TemporaryDirectory() as work:
            path = self._wout_path(eq, Path(work))
            values = np.array([self._one_surface(path, float(s), Path(work))
                               for s in self.surfaces], dtype=float)  # fmt: skip
        self._cache = (eq, values)
        return values

    @staticmethod
    def _wout_path(eq: Any, work: Path) -> Path:
        """A wout on disk for ``eq``, written only when it is not already a file."""
        if isinstance(eq, (str, Path)):
            return Path(eq)
        from vmex import write_wout  # noqa: PLC0415

        return Path(write_wout(work / "wout_objective.nc", getattr(eq, "wout", eq)))

    def _one_surface(self, equilibrium: Path, s: float, work: Path) -> float:
        import warnings  # noqa: PLC0415

        from dkx.api import batched_er_scan  # noqa: PLC0415
        from dkx.run import run_profile  # noqa: PLC0415

        deck = work / f"in_{s:.6f}.namelist"
        a_hat = _minor_radius(equilibrium)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.ambipolar:
                    er = np.asarray(self.er_values, dtype=float)
                    deck.write_text(self.namelist(equilibrium, s, er=0.0, a_hat=a_hat))
                    scan = batched_er_scan(deck, er)
                    current = np.asarray(scan.radial_current, dtype=float).ravel()
                    root = self._ion_root(er, current)
                    if root is None:
                        return float("nan")
                    j_par = np.asarray(scan.moments["FSABjHat"], dtype=float).ravel()
                    order = np.argsort(er)
                    value = float(np.interp(root, er[order], j_par[order]))
                else:
                    deck.write_text(
                        self.namelist(equilibrium, s, er=self.er_kV_per_m, a_hat=a_hat))
                    # emit=None: run_profile prints a per-run banner by default,
                    # and an optimization makes hundreds of these calls.
                    run = run_profile(deck, emit=None)
                    value = float(np.asarray(run.moments["FSABjHat"]).reshape(()))
        except Exception as exc:  # pragma: no cover - equilibrium-dependent
            if self.verbose:
                print(f"  {self.name}: s={s:.3f} unavailable ({type(exc).__name__}: {exc})")
            return float("nan")
        return value * PARALLEL_CURRENT

    @staticmethod
    def _ion_root(er: np.ndarray, radial_current: np.ndarray) -> float | None:
        """Most negative bracketed zero of ``J_r(E_r)``, by linear interpolation."""
        order = np.argsort(er)
        x, y = er[order], radial_current[order]
        for i in range(x.size - 1):
            if y[i] == 0.0:
                return float(x[i])
            if y[i] * y[i + 1] < 0.0:
                return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
        return None



#: Kinetic grid of :class:`KineticBootstrapMismatch`: the coarsest grid on which
#: the Boozer route agrees in sign with the VMEC-file route and with Redl
#: (``tests/test_boozer_route_sign.py``; coarser than ``Nxi = 16, Nx = 4`` the
#: pitch-angle-scattering current changes sign).  A teaching grid, not a
#: converged one: refine ``Nxi`` first at low collisionality.
DEFAULT_TRACED_RESOLUTION: dict[str, int] = {"Ntheta": 11, "Nzeta": 11, "Nxi": 16, "NL": 4, "Nx": 4}

class KineticBootstrapMismatch:
    r"""DKX ``<j.B>`` as a traceable VMEX objective term, the kinetic ``RedlBootstrapMismatch``.

    The interface is that of ``vmex.core.bootstrap.RedlBootstrapMismatch``:
    ``residuals_state(state, runtime)`` returns the residual vector and
    ``total(state, runtime)`` its sum of squares, so an instance goes into a
    ``vmex.optimize`` objective tuple as ``(term, 0.0, weight)`` and VMEX's
    implicit Jacobian differentiates it.  On each surface the chain is traced
    end to end: VMEX state -> ``boozer_input_tables`` -> ``booz_xform_jax``
    -> :func:`~dkx.workflows.geometry_adapters.kinetic_operator_on_boozer_surface`
    -> exact structured drift-kinetic solve -> ``FSABjHat``.

    Residuals, with :math:`J_v` the equilibrium's own ``<j.B>``
    (``vmex.core.bootstrap.vmec_j_dot_B``) and :math:`J_k` the kinetic one:

    - ``mismatch=True`` (default): Redl's self-normalized form
      :math:`R_i = (J_{v,i} - J_{k,i}) / \sqrt{\sum_j (J_{v,j} + J_{k,j})^2}`,
      so ``total`` is bounded by 1 and weights compare with Redl's ``f_boot``;
    - ``mismatch=False``: :math:`R_i = J_{k,i} / 10^6`, the kinetic current in
      MA T/m\ :sup:`2`, for a tuple that targets ``<j.B>`` itself (``0.0`` for
      a current-free design).

    ``profiles`` carries ``ne_coeffs`` [m^-3], ``Te_coeffs`` and ``Ti_coeffs``
    [eV], polynomials in ``s`` lowest order first; ``KineticProfiles`` of VMEX
    is exactly that, so one object feeds this term and Redl's.  Ions are
    hydrogen with ``n_i = n_e``.

    Each requested surface is moved to the nearest half-mesh row of the
    equilibrium, where VMEX's field tables live; :meth:`current_profiles`
    reports the row values.  The per-row operator templates and the Boozer
    plan are built on the host the first time a radial grid and mode set are
    seen and cached, so a traced call is pure arithmetic.

    **Collision operator.**  The default, pitch-angle scattering
    (``collision_operator=1``), is solved by the exact structured route, so a
    residual and its derivative are cheap.  It has no momentum-restoring
    term, and ``<j.B>`` is a parallel-momentum moment: it overestimates Redl
    by 35-47% on a precise-QA equilibrium (see ``DEFAULT_COLLISION_OPERATOR``).
    ``collision_operator=0`` (Fokker-Planck) conserves momentum and costs a
    Krylov solve.

    **Handedness.**  The Boozer route converts VMEC's handedness through
    ``psiAHat = |psi_a| signgs sign(G + iota I)``
    (:func:`~dkx.workflows.geometry_adapters.boozer_route_psi_a_hat`),
    evaluated traced from the runtime's flux and the surface's Boozer
    ``G``, ``I`` and ``iota``.

    ``er_kV_per_m`` is a prescribed radial electric field; ``minor_radius``
    [m] defines the radius it is a derivative with respect to and matters only
    when ``er_kV_per_m != 0``.
    """

    name = "j_boot_dkx"

    def __init__(
        self,
        profiles: Any,
        surfaces: Sequence[float] = (0.25, 0.5, 0.75),
        *,
        mismatch: bool = True,
        resolution: dict[str, int] | None = None,
        collision_operator: int = 1,
        er_kV_per_m: float = 0.0,
        minor_radius: float = 1.0,
        mboz: int = 6,
        nboz: int = 6,
        tol: float = 1.0e-10,
    ) -> None:
        self.profiles = profiles
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        if np.any((self.surfaces <= 0.0) | (self.surfaces >= 1.0)):
            raise ValueError("surfaces must lie strictly inside (0, 1)")
        self.mismatch = bool(mismatch)
        self.resolution = dict(DEFAULT_TRACED_RESOLUTION if resolution is None else resolution)
        self.collision_operator = int(collision_operator)
        self.er_kV_per_m = float(er_kV_per_m)
        self.minor_radius = float(minor_radius)
        self.mboz, self.nboz = int(mboz), int(nboz)
        self.tol = float(tol)
        self._templates: dict[int, tuple[np.ndarray, np.ndarray, list[Any]]] = {}
        self._plans: dict[tuple[Any, ...], Any] = {}

    # -- host-side set-up, cached ---------------------------------------------

    def rows(self, ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Half-mesh rows nearest to ``surfaces`` on an ``ns``-point grid, and their ``s``."""
        rows = np.clip(np.rint(self.surfaces * (ns - 1) + 0.5).astype(int), 1, ns - 1)
        rows = np.unique(rows)
        return rows, (rows - 0.5) / (ns - 1)

    def _template(self, s: float) -> Any:
        """Operator at ``psiAHat = 1``: its gradient leaves hold ``d/ds`` values."""
        from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: PLC0415
        from dkx.inputs import SfincsInput  # noqa: PLC0415
        from dkx.namelist import parse_sfincs_input_text  # noqa: PLC0415

        ne, dne = _polynomial(self.profiles.ne_coeffs, s)
        te, dte = _polynomial(self.profiles.Te_coeffs, s)
        ti, dti = _polynomial(self.profiles.Ti_coeffs, s)
        to_r_hat = 2.0 * float(np.sqrt(s)) / self.minor_radius  # d/drHat = (2 sqrt(s)/a) d/ds
        deck = SfincsInput.from_params(
            geometryScheme=1, psiAHat=1.0, aHat=self.minor_radius,  # geometry replaced per call
            inputRadialCoordinate=1, inputRadialCoordinateForGradients=4, psiN_wish=float(s),
            Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4], nHats=[ne / 1e20] * 2,
            THats=[ti / 1e3, te / 1e3], dNHatdrHats=[dne / 1e20 * to_r_hat] * 2,
            dTHatdrHats=[dti / 1e3 * to_r_hat, dte / 1e3 * to_r_hat],
            collisionOperator=self.collision_operator, Delta=4.5694e-3, alpha=1.0,
            nu_n=8.330e-3, Er=self.er_kV_per_m, **self.resolution,
        )  # fmt: skip
        return kinetic_operator_from_namelist(parse_sfincs_input_text(deck.to_namelist()))

    def _setup(self, ns: int) -> tuple[np.ndarray, np.ndarray, list[Any]]:
        if ns not in self._templates:
            rows, s_rows = self.rows(ns)
            self._templates[ns] = (rows, s_rows, [self._template(float(s)) for s in s_rows])
        return self._templates[ns]

    def _plan(self, xm: np.ndarray, xn: np.ndarray, nfp: int) -> Any:
        from booz_xform_jax.jax_api import BoozerConfig, prepare_booz_xform_plan  # noqa: PLC0415

        key = (nfp, xm.tobytes(), xn.tobytes())
        if key not in self._plans:
            self._plans[key] = prepare_booz_xform_plan(
                nfp=nfp, asym=False, xm=xm, xn=xn, xm_nyq=xm, xn_nyq=xn,
                config=BoozerConfig.from_env(mboz=self.mboz, nboz=self.nboz))  # fmt: skip
        return self._plans[key]

    # -- traced evaluation ----------------------------------------------------

    def _kinetic(self, state: Any, rt: Any) -> tuple[np.ndarray, Any]:
        """``(s_rows, <j.B>_kinetic)`` [A T/m^2], traced in ``state``."""
        import jax.numpy as jnp  # noqa: PLC0415
        from booz_xform_jax.jax_api import booz_xform_jax_impl  # noqa: PLC0415
        from vmex.core.boozer_tables import boozer_input_tables  # noqa: PLC0415

        from dkx.run import profile_moments_from_operator  # noqa: PLC0415
        from dkx.solve import solve  # noqa: PLC0415
        from dkx.workflows.geometry_adapters import (  # noqa: PLC0415
            PSI_A_HAT_LEAVES,
            kinetic_operator_on_boozer_surface,
        )

        setup = rt.setup
        s_full = np.asarray(setup.s_full)
        nfp = int(rt.resolution.nfp)
        rows, s_rows, templates = self._setup(int(s_full.size))
        # |psi_a| = |phi_edge|/(2 pi) in Wb/rad, as VMEX's bootstrap lane forms it.
        psi_a = jnp.abs((s_full[1] - s_full[0]) * jnp.sum(jnp.asarray(setup.phipf)[1:]))
        currents = []
        for row, template in zip(rows, templates):
            tables = boozer_input_tables(state, rt, int(row))
            xm, xn = np.asarray(tables["xm"]), np.asarray(tables["xn"])
            plan = self._plan(xm, xn, nfp)
            booz = booz_xform_jax_impl(
                **{k: jnp.asarray(tables[k])[None] for k in (
                    "rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota")},
                xm=jnp.asarray(xm), xn=jnp.asarray(xn), xm_nyq=jnp.asarray(xm),
                xn_nyq=jnp.asarray(xn), constants=plan.constants, grids=plan.grids, plan=plan,
            )  # fmt: skip
            g, i, iota = booz["bvco_b"][0], booz["buco_b"][0], booz["iota_b"][0]
            # The template holds d/ds gradients (psiAHat = 1); dividing by
            # |psi_a| and converting the handedness gives the signed psiAHat
            # of boozer_route_psi_a_hat, traced.
            operator = kinetic_operator_on_boozer_surface(
                replace(template, **{leaf: getattr(template, leaf) / psi_a
                                     for leaf in PSI_A_HAT_LEAVES}),
                bmnc_b=booz["bmnc_b"][0], ixm_b=np.asarray(plan.grids.xm_b),
                ixn_b=np.asarray(plan.grids.xn_b), nfp=nfp, iota=iota, g_hat=g, i_hat=i,
                signgs=int(np.sign(setup.signgs)))  # fmt: skip
            rhs = operator.rhs()
            # tier1_keep_lowest = Nxi keeps every Legendre block, so the
            # solution satisfies the original equation, not a truncation.
            solved = solve(operator, rhs, method="auto", tol=self.tol, differentiable=True,
                           tier1_keep_lowest=operator.n_xi, emit=None)  # fmt: skip
            moments = profile_moments_from_operator(operator, solved.x.reshape(-1))
            currents.append(moments["FSABjHat"].reshape(()) * PARALLEL_CURRENT)
        return s_rows, jnp.stack(currents)

    def current_profiles(self, state: Any, rt: Any) -> tuple[np.ndarray, Any, Any]:
        """``(s, <j.B>_VMEX, <j.B>_DKX)`` in A T/m^2 on the kinetic rows."""
        from vmex.core.bootstrap import vmec_j_dot_B  # noqa: PLC0415

        s_rows, kinetic = self._kinetic(state, rt)
        return s_rows, vmec_j_dot_B(state, rt, surfaces=s_rows), kinetic

    def residuals_state(self, state: Any, rt: Any) -> Any:
        """Traceable residual vector; ``sum(r**2) = total``."""
        import jax.numpy as jnp  # noqa: PLC0415

        if not self.mismatch:
            return self._kinetic(state, rt)[1] / DEFAULT_REFERENCE_CURRENT
        _, jv, jk = self.current_profiles(state, rt)
        denominator = jnp.sum((jv + jk) ** 2)
        return (jv - jk) / jnp.sqrt(jnp.maximum(denominator, jnp.finfo(denominator.dtype).tiny))

    def total(self, state: Any, rt: Any) -> Any:
        """Scalar objective ``sum(residuals**2)``, for ``EquilibriumReporter``."""
        import jax.numpy as jnp  # noqa: PLC0415

        r = self.residuals_state(state, rt)
        return jnp.sum(r * r)

    __call__ = residuals_state


def plot_kinetic_bootstrap_current(
    path: str | Path,
    equilibrium: Any,
    kinetic: KineticBootstrapCurrent,
    *,
    redl: Any = None,
    dpi: int = 150,
) -> Path:
    """Overlay the kinetic, equilibrium and (optionally) Redl ``<J.B>`` profiles.

    The counterpart of ``vmex.plot_bootstrap_current`` for this term, in the
    same units (MA T/m^2) so the two figures can be read side by side.  ``redl``
    is a ``vmex.core.bootstrap.RedlBootstrapMismatch``; passing it draws the
    analytic profile the kinetic one is meant to replace, which is the
    comparison a reader of a DKX-driven optimization will want first.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    surfaces, kinetic_current = kinetic.current_profile(equilibrium)
    kinetic_current = np.asarray(kinetic_current, dtype=float) / 1.0e6
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.plot(surfaces, kinetic_current, "o-", label="DKX kinetic")
    if redl is not None:
        s_redl, equilibrium_current, redl_current = redl.current_profiles(equilibrium)
        axis.plot(np.asarray(s_redl, dtype=float),
                  np.asarray(equilibrium_current, dtype=float) / 1.0e6,
                  "^:", label="VMEC equilibrium")  # fmt: skip
        axis.plot(np.asarray(s_redl, dtype=float),
                  np.asarray(redl_current, dtype=float) / 1.0e6,
                  "s--", label="Redl bootstrap")  # fmt: skip
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.set(xlabel=r"$s=\psi/\psi_{\rm edge}$",
             ylabel=r"$\langle\mathbf{J}\!\cdot\!\mathbf{B}\rangle$ [MA T m$^{-2}$]")  # fmt: skip
    finite = kinetic_current[np.isfinite(kinetic_current)]
    if finite.size:
        axis.text(0.03, 0.95,
                  f"kinetic RMS = {float(np.sqrt(np.mean(finite**2))):.3g} MA T m$^{{-2}}$",
                  transform=axis.transAxes, fontsize=9, va="top",
                  bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85})  # fmt: skip
    axis.legend(frameon=True, loc="best")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=int(dpi))
    plt.close(figure)
    return path.resolve()
