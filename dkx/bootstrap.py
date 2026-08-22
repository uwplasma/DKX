"""Kinetic bootstrap current as an objective term on a VMEX equilibrium.

``vmex.core.bootstrap.RedlBootstrapMismatch`` drives a VMEC equilibrium toward
the Redl *analytic* bootstrap current.  This module is the drift-kinetic
counterpart: DKX solves the linearized drift-kinetic equation on the same
equilibrium and returns :math:`\\langle j_\\parallel B\\rangle` itself, so an
optimizer can drive the kinetic bootstrap current toward zero rather than match
a formula.  The two are complementary, not redundant --- Redl is a fit valid in
the banana regime for a quasisymmetric field, and the kinetic current is what
the device would actually carry.

The term presents exactly the interface ``vmex.core.optimize`` expects of a
wout-lane objective: a single-argument callable returning a residual vector,
with ``total`` for the reporter.  Adding it to an existing
``vmex/examples/optimization`` script is one import and one tuple::

    from dkx.bootstrap import KineticBootstrapCurrent

    kinetic = KineticBootstrapCurrent(profiles, surfaces=SURFACES)
    objective_function_terms.append((kinetic, 0.0, KINETIC_WEIGHT))

DKX is a host code rather than a traced one, so a problem carrying this term
must be built with ``derivative_method="finite_difference"``; the traceable
``residuals_state`` lane belongs to the Redl term.  Each residual evaluation
costs one DKX solve per surface (per ``E_r`` point when ``ambipolar=True``), so
keep ``surfaces`` short --- this is minutes per objective evaluation, not
milliseconds.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dkx.units import PARALLEL_CURRENT

__all__ = ["DEFAULT_ER_BRACKET", "DEFAULT_RESOLUTION", "KineticBootstrapCurrent",
           "plot_kinetic_bootstrap_current"]  # fmt: skip

#: Grid for one bootstrap-current solve.  Nxi >= Nzeta because the Legendre
#: resolution is what limits accuracy at low collisionality; Nx=5 is the
#: smallest speed grid that keeps the current converged for these profiles.
DEFAULT_RESOLUTION: dict[str, int] = {"n_theta": 21, "n_zeta": 31, "n_xi": 32, "n_x": 5}

#: ``E_r`` values (kV/m) bracketing the ion root, used only when the ambipolar
#: root is requested.
DEFAULT_ER_BRACKET: tuple[float, ...] = (-8.0, -4.0, -2.0, -1.0, -0.4, 0.4, 1.0, 2.0)

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
  inputRadialCoordinateForGradients = 1
  rN_wish = {r_n:.10g}
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = {n_hat:.10g} {n_hat:.10g}
  THats = {ti_hat:.10g} {te_hat:.10g}
  dNHatdpsiNs = {dn_ds:.10g} {dn_ds:.10g}
  dTHatdpsiNs = {dti_ds:.10g} {dte_ds:.10g}
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
    collision_operator: int = 1
    verbose: bool = False

    name = "j_boot_dkx"

    def __post_init__(self) -> None:
        self.surfaces = np.atleast_1d(np.asarray(self.surfaces, dtype=float))
        self._cache: tuple[Any, np.ndarray] | None = None

    # -- plasma and deck ------------------------------------------------------

    def plasma_at(self, s: float) -> dict[str, float]:
        """The ``&speciesParameters`` entries for surface ``s``.

        SFINCS normalizes to ``nBar = 1e20 m^-3`` and ``TBar = 1 keV``
        (:mod:`dkx.units`), and ``inputRadialCoordinateForGradients = 1`` makes
        the gradients ``d/d psiN`` with ``psiN = s`` --- the coordinate the
        profile polynomials are already written in, so no chain rule is needed.
        """
        ne, dne = _polynomial(self.profiles.ne_coeffs, s)
        te, dte = _polynomial(self.profiles.Te_coeffs, s)
        ti, dti = _polynomial(self.profiles.Ti_coeffs, s)
        return {
            "n_hat": ne / 1.0e20, "dn_ds": dne / 1.0e20,
            "te_hat": te / 1.0e3, "dte_ds": dte / 1.0e3,
            "ti_hat": ti / 1.0e3, "dti_ds": dti / 1.0e3,
        }  # fmt: skip

    def namelist(self, equilibrium: str | Path, s: float, *, er: float) -> str:
        """The full DKX input deck for one surface of one equilibrium."""
        return _TEMPLATE.format(
            equilibrium=str(equilibrium), r_n=float(np.sqrt(max(s, 0.0))), er=float(er),
            collision_operator=int(self.collision_operator),
            **self.plasma_at(float(s)), **self.resolution,
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
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.ambipolar:
                    er = np.asarray(self.er_values, dtype=float)
                    deck.write_text(self.namelist(equilibrium, s, er=0.0))
                    scan = batched_er_scan(deck, er)
                    current = np.asarray(scan.radial_current, dtype=float).ravel()
                    root = self._ion_root(er, current)
                    if root is None:
                        return float("nan")
                    j_par = np.asarray(scan.moments["FSABjHat"], dtype=float).ravel()
                    order = np.argsort(er)
                    value = float(np.interp(root, er[order], j_par[order]))
                else:
                    deck.write_text(self.namelist(equilibrium, s, er=self.er_kV_per_m))
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
