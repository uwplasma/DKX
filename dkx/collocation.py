r"""Pitch-**collocation** discretization of the radially-local drift-kinetic equation.

A second discretization of the same continuum physics :mod:`dkx.drift_kinetic`
already solves, and a solver route the modal one structurally cannot have.

Why a second pitch basis exists
-------------------------------
In dkx's Legendre-*modal* basis the state holds coefficients ``f_l``, and
parallel streaming ``x xi b.grad`` together with the mirror force couple
``l -> l +- 1`` while contributing **nothing** to the ``(l, l)`` block.  Every
line of an alternating line relaxation therefore misses the operator's dominant
term.  Measured on dkx's own W7-X operator, exact alternating line solves have
spectral radius ``5.9e6`` and a two-grid factor of ``4.0e13``: the modal basis
has no convergent relaxation at any collisionality, so no multigrid cycle can
be built on it, whatever the transfers or the cycle shape.

On a pitch-*angle* grid ``xi`` is a multiplication operator, so streaming is
diagonal in pitch and an upwindable advection in the angles; the mirror force
becomes an advection in ``alpha`` whose upwind form is diagonally dominant in
turn; and pitch-angle scattering becomes a local tridiagonal diffusion with a
positive diagonal.  All of ``(alpha, theta, zeta)`` can then be coarsened
together, keeping the advection direction fixed relative to the mesh on every
level.  Same operator, same geometry, first-order upwind: two-grid factor
``0.24``.

What is discretized
-------------------
One species, radially local, DKES-like trajectories.  Per speed node ``x``,

.. math::

    K f = a \left[ \xi\, \mathbf{b}\cdot\nabla f
                   - \tfrac{1}{2}(1-\xi^2)\,
                     (\mathbf{b}\cdot\nabla \ln B)\, \partial_\xi f \right]
          + w^{E}_\theta \partial_\theta f + w^{E}_\zeta \partial_\zeta f
          - \frac{\hat\nu_D}{2}\, \partial_\xi\!\left[(1-\xi^2)\,\partial_\xi f\right],

with ``a = x sqrt(THat/mHat)``, all quantities in the SFINCS v3 "Hat"
normalization (:mod:`dkx.drift_kinetic`), and ``w^E`` the ``ExB`` coefficients
of :meth:`dkx.drift_kinetic.KineticOperator._exb_coefficients`.  This is
``collisionOperator = 1``, ``includeXDotTerm = .false.``,
``includeElectricFieldTermInXiDot = .false.``, ``includePhi1 = .false.``, so the
two discretizations solve the *same* boundary-value problem and must agree in
the continuum limit.  The ``RHSMode = 1`` drive of ``evaluateResidual.F90`` is
carried over exactly: in pitch its gradient part is ``4/3 P_0 + 2/3 P_2 =
1 + xi^2`` and its inductive part is ``P_1 = xi``.

Grids
-----
``alpha``
    Uniform **half-index** angle grid, ``alpha_j = pi (2j+1)/(2 Nalpha)``,
    ``xi = cos alpha``, spacing ``h = pi/Nalpha``.  Half-index placement keeps
    ``xi = +-1`` -- where the mirror speed vanishes and the collision operator
    is singular -- off the grid; the collision fluxes then live on the integer
    half-points where ``sin`` vanishes identically, so the no-flux condition is
    *exact* rather than imposed.  Regularity of ``f`` in the full velocity
    space forces ``df/dalpha -> 0`` at ``alpha = 0, pi``, which the
    index-clamped stencil closure realizes.  A half-index grid is also the one a
    factor-two coarsening maps onto itself (``solvax.transfer`` boundary
    ``"reflective"``).
``theta`` / ``zeta``
    Uniform periodic grids over ``[0, 2 pi)`` and ``[0, 2 pi/Nperiods)``, and
    deliberately **even**-length.  SFINCS forces them odd only because a
    *centered* periodic first-derivative matrix of even length has the Nyquist
    mode in its null space; an upwind stencil does not, and even lengths are
    what factor-two periodic coarsening needs.
``x``
    dkx's existing speed grid, untouched and never coarsened.  Pitch-angle
    scattering leaves the operator block diagonal in ``x``, and the dense
    collision operators of a later phase would make a coarse ``x`` grid a
    different quadrature rule rather than a coarser stencil -- no local smoother
    is complementary to that.

Stencils
--------
Streaming and ``ExB`` share one wind per angular axis, the mirror force is the
wind in ``alpha``, and each is upwinded with a stencil dkx already carries
(:data:`COLLOCATION_STENCILS`, built on
:func:`dkx.phase_space.widened_upwind_stencil`).  The widened stencils skip near
neighbours to hold diagonal weight at a given formal order; their diagonal
dominance ``|c_0| / sum_{j != 0} |c_j|`` is ``1`` (``"up1"``), ``5/7``
(``"up3"``) and ``13/21`` (``"up4"``), and the two-grid factor is a monotone
function of it.  Selecting a widened ``stencil`` with ``relaxation_stencil =
"up1"`` is Brandt's double discretization: an accurate fine-level operator with
an upwinded smoother and coarse levels.

Collisions use the conservative flux form

.. math::

    (Cf)_j = -\frac{\hat\nu_D}{2\,\sin\alpha_j\,h^2}
             \left[ s_{j+1/2}(f_{j+1}-f_j) - s_{j-1/2}(f_j-f_{j-1}) \right],
    \qquad s_{j\pm 1/2} = \sin(\alpha_j \pm h/2),

which is tridiagonal, has a positive diagonal, annihilates constants exactly,
and needs no boundary closure.

Solvability without bordering
-----------------------------
``K`` has a one-dimensional null space per speed node -- the state constant in
``(alpha, theta, zeta)`` -- exactly as the continuum operator does, and visibly
so in the discrete rows (every stencil row sums to zero).  SFINCS closes the
system with ``constraintScheme = 2``: one pitch-constant particle source per
speed node plus one constraint that the flux-surface and pitch average of ``f``
vanish, which *borders* the matrix.  This module reaches the same solution with
the rank-one shift

.. math::  M = K + \sigma\, e\, m^\top ,

``e`` the constant state, ``m`` the normalized constraint functional and
``sigma`` the mean diagonal of ``K``.  Because ``m^\top e = 1`` and the left
null vector of ``K`` is not orthogonal to ``e``, ``M`` is nonsingular, and
``M x = b`` yields ``K x = b - e s`` with the *same* source amplitude the
bordered system produces, up to a multiple of ``e`` that
:meth:`CollocationOperator.project` removes.  Keeping the system square and
unbordered is what lets a local relaxation and a rediscretized coarse operator
see the same problem on every level -- a bordered row has no coarse-grid
analogue.

References
----------
- M. Landreman, H. M. Smith, A. Mollen & P. Helander, "Comparison of particle
  trajectories and collision operators for collisional transport in
  nonaxisymmetric plasmas", Phys. Plasmas **21**, 042503 (2014) -- the
  drift-kinetic equation, normalizations and trajectory options solved here.
- A. Brandt, "Multi-level adaptive solutions to boundary-value problems",
  Math. Comp. **31**, 333 (1977) -- smoothing/coarse-grid complementarity and
  (section 3) the upwind requirement for advection.
- A. Brandt, "Multigrid solvers for non-elliptic and singular-perturbation
  steady-state problems", Weizmann Institute (1981) -- double discretization.
- U. Trottenberg, C. W. Oosterlee & A. Schueller, *Multigrid*, Academic Press
  (2001) -- ch. 2 (transfers, rediscretized coarse operators), ch. 5
  (semicoarsening, line relaxation), section 7.4 (convection-dominated cycles).
- B. P. Leonard, "A stable and accurate convective modelling procedure based on
  quadratic upstream interpolation", Comput. Methods Appl. Mech. Eng. **19**, 59
  (1979) -- the accuracy/diagonal-weight trade the widened stencils make.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dkx.api import SolverOptions
from dkx.drift_kinetic import _geometry_and_radial, kinetic_operator_build_from_namelist
from dkx.magnetic_geometry import FluxSurfaceGeometry
from dkx.moments import FluxSurface, SpeciesParams, StateLayout, VelocityGrid, vm_flux_moments
from dkx.phase_space import widened_upwind_stencil

__all__ = [
    "COLLOCATION_STENCILS",
    "CollocationOperator",
    "CollocationOptions",
    "CollocationSolution",
    "collocation_operator",
    "collocation_operator_from_namelist",
    "solve_collocation",
]

#: Upwind first-derivative stencils in the positive-wind (upstream-leaning)
#: orientation, ``key -> (offsets, coefficients)`` with coefficients in units of
#: ``1/h``.  ``"up1"`` is the two-point first-order stencil; ``"up3"``/``"up4"``
#: are dkx's widened third-/fourth-order stencils
#: (:func:`dkx.phase_space.widened_upwind_stencil`, namelist codes ``+-103`` and
#: ``+-104``).
COLLOCATION_STENCILS: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {
    "up1": ((-1, 0), (-1.0, 1.0)),
    "up3": widened_upwind_stencil(order=3),
    "up4": widened_upwind_stencil(order=4),
}


def _mirrored(
    offsets: Sequence[int], coefficients: Sequence[float]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Reflect a stencil about its node: the negative-wind orientation."""
    pairs = sorted(zip((-o for o in offsets), (-c for c in coefficients)))
    return tuple(int(o) for o, _ in pairs), tuple(float(c) for _, c in pairs)


def _diff_matrices(
    *, n: int, spacing: float, stencil: str, periodic: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Upwind ``d/dx`` matrices for a positive and a negative wind.

    Both have zero row sums, so a constant is annihilated exactly and the
    discrete operator inherits the continuum null space.  Non-periodic axes
    clamp out-of-range indices, which is the zero-gradient closure the pitch
    axis needs (see the module docstring).
    """
    offsets, coefficients = COLLOCATION_STENCILS[stencil]
    span = max(offsets) - min(offsets)
    if periodic and n <= span:
        raise ValueError(f"a periodic axis needs n > {span} for stencil {stencil!r}, got n={n}")
    rows = np.arange(int(n))

    def build(offs: Sequence[int], coefs: Sequence[float]) -> np.ndarray:
        out = np.zeros((int(n), int(n)), dtype=np.float64)
        for offset, coefficient in zip(offs, coefs):
            shifted = rows + int(offset)
            cols = np.mod(shifted, n) if periodic else np.clip(shifted, 0, n - 1)
            np.add.at(out, (rows, cols), float(coefficient) / float(spacing))
        return out

    return build(offsets, coefficients), build(*_mirrored(offsets, coefficients))


def _band_weights(stencil: str) -> tuple[dict[int, float], dict[int, float]]:
    """Tridiagonal part of a stencil, for a positive and a negative wind."""
    offsets, coefficients = COLLOCATION_STENCILS[stencil]
    forward = dict(zip(offsets, coefficients))
    backward = dict(zip(*_mirrored(offsets, coefficients)))
    return (
        {k: forward.get(k, 0.0) for k in (-1, 0, 1)},
        {k: backward.get(k, 0.0) for k in (-1, 0, 1)},
    )


@dataclass(frozen=True)
class CollocationOptions:
    """Typed, composable knobs for the pitch-collocation backend.

    Pass alongside :class:`dkx.api.SolverOptions` to :func:`solve_collocation`:
    ``SolverOptions`` owns the outer Krylov (tolerance, cycle size, recycling)
    and this owns the discretization and the multigrid preconditioner, so the
    two surfaces stay orthogonal.  Every field is an explicit API argument --
    no environment variables, and no hidden defaults that change the physics.

    Attributes:
        n_alpha: Pitch-angle resolution.  Even, so the factor-two cell-centered
            coarsening is exact.
        n_theta: Poloidal resolution.  Even (see the module docstring).
        n_zeta: Toroidal resolution per field period.  Even.
        stencil: Upwind stencil of the operator itself, a key of
            :data:`COLLOCATION_STENCILS`.  ``"up1"`` is the most robust,
            ``"up3"``/``"up4"`` are more accurate per point.
        relaxation_stencil: Stencil the smoother's line bands are built from.
            Leaving it at ``"up1"`` under a widened ``stencil`` is double
            discretization (Brandt 1981).
        smoother: ``"plane"`` solves each ``(theta, zeta)`` plane exactly and
            then relaxes the pitch lines; ``"line"`` alternates exact line
            solves in all three axes; ``"upwind"`` replaces those by
            wind-ordered sweeps (Brandt & Yavneh 1993).  Parallel streaming
            lives entirely in the angular plane and is not aligned with either
            angle, so ``"plane"`` is the only one of the three that stays a
            smoother as the collisionality falls -- see :func:`_smoother`.
        cycle: ``"v"``, ``"w"`` or ``"f"``.
        levels: Maximum number of coarsening steps.
        pre_smooth: Pre-smoothing sweeps per level.
        post_smooth: Post-smoothing sweeps per level.
        omega: Relaxation weight of each sweep.  ``1`` -- an undamped exact
            block solve -- is measured to be right across all seven decades of
            mesh Peclet number the speed grid spans, once the null-space shift
            is kept out of the cycle (:func:`_preconditioner`).
        min_coarse: Smallest length a coarsened axis may reach.
        preconditioner: ``"multigrid"``, or ``"none"`` to measure the
            unpreconditioned operator.
        species: Index of the species to solve for.
    """

    n_alpha: int = 32
    n_theta: int = 16
    n_zeta: int = 16
    stencil: str = "up1"
    relaxation_stencil: str = "up1"
    smoother: str = "line"
    cycle: str = "v"
    levels: int = 3
    pre_smooth: int = 1
    post_smooth: int = 1
    omega: float = 1.0
    min_coarse: int = 4
    preconditioner: str = "multigrid"
    species: int = 0

    def __post_init__(self) -> None:
        for name in ("n_alpha", "n_theta", "n_zeta"):
            value = int(getattr(self, name))
            if value < 4 or value % 2:
                raise ValueError(f"{name} must be an even integer >= 4, got {value}")
        for name in ("stencil", "relaxation_stencil"):
            value = getattr(self, name)
            if value not in COLLOCATION_STENCILS:
                raise ValueError(
                    f"unknown {name} {value!r}; expected one of {sorted(COLLOCATION_STENCILS)}"
                )
        if self.smoother not in ("plane", "line", "upwind"):
            raise ValueError(
                f"unknown smoother {self.smoother!r}; expected 'plane', 'line' or 'upwind'"
            )
        if self.preconditioner not in ("multigrid", "none"):
            raise ValueError(
                f"unknown preconditioner {self.preconditioner!r}; expected 'multigrid' or 'none'"
            )

    @property
    def grid(self) -> tuple[int, int, int]:
        """``(n_alpha, n_theta, n_zeta)``."""
        return (int(self.n_alpha), int(self.n_theta), int(self.n_zeta))

    def refined(self, factor: int) -> "CollocationOptions":
        """The same settings on a grid refined by ``factor`` in every coarsened axis."""
        return replace(
            self,
            n_alpha=int(self.n_alpha * factor),
            n_theta=int(self.n_theta * factor),
            n_zeta=int(self.n_zeta * factor),
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CollocationOperator:
    """Radially-local DKE for one species on an ``(x, alpha, theta, zeta)`` grid.

    States are plain ``(n_x, n_alpha, n_theta, n_zeta)`` arrays: there are no
    bordered constraint rows (module docstring, *Solvability*).  The class is a
    pytree of JAX arrays and :meth:`apply` is jit-able and differentiable with
    respect to every stored coefficient, so the backend is usable inside
    :mod:`dkx.sensitivity`-style gradient work without further plumbing.

    Attributes:
        x: Speed nodes ``v/v_th``, ``(X,)``.
        x_weights: Speed quadrature weights, ``(X,)``.
        alpha: Pitch-angle nodes in ``(0, pi)``, ``(A,)``.
        xi: ``cos(alpha)``, ``(A,)``.
        theta: Poloidal nodes, ``(T,)``.
        zeta: Toroidal nodes of one field period, ``(Z,)``.
        d_alpha: Stacked ``(positive, negative)`` wind pitch derivative
            matrices, ``(2, A, A)``.
        d_theta: Stacked poloidal derivative matrices, ``(2, T, T)``.
        d_zeta: Stacked toroidal derivative matrices, ``(2, Z, Z)``.
        w_alpha: Mirror advection ``dalpha/dt``, ``(X, A, T, Z)``.
        w_theta: Streaming plus ``ExB`` poloidal wind, ``(X, A, T, Z)``.
        w_zeta: Streaming plus ``ExB`` toroidal wind, ``(X, A, T, Z)``.
        collision: Stacked ``(lower, diag, upper)`` pitch bands, ``(3, X, A)``.
        spacing: ``(h_alpha, h_theta, h_zeta)``.
        pin_weight: Normalized constraint functional ``m``, ``(A, T, Z)``.
        pin_sigma: Rank-one shift strength per speed node, ``(X,)``.
        drive_radial: ``RHSMode=1`` gradient drive amplitude, ``(X, T, Z)``.
        drive_epar: Inductive drive amplitude, ``(X, T, Z)``.
        theta_weights: Poloidal flux-surface-average weights, ``(T,)``.
        zeta_weights: Toroidal flux-surface-average weights, ``(Z,)``.
        pitch_weights: Midpoint pitch quadrature ``h sin(alpha)``, ``(A,)``.
        surface: Geometry arrays the moment integrals need.
        species_params: Charge, mass, temperature and density of the species.
        delta: Namelist ``Delta`` (reference gyroradius over reference length).
        normalization_alpha: Namelist ``alpha`` (``e phiBar / TBar``).
        n_x, n_alpha, n_theta, n_zeta, n_periods: Grid sizes (static).
        stencil: Upwind stencil of the operator (static).
        relaxation_stencil: Upwind stencil of the relaxation bands (static).
    """

    x: jnp.ndarray
    x_weights: jnp.ndarray
    alpha: jnp.ndarray
    xi: jnp.ndarray
    theta: jnp.ndarray
    zeta: jnp.ndarray
    d_alpha: jnp.ndarray
    d_theta: jnp.ndarray
    d_zeta: jnp.ndarray
    w_alpha: jnp.ndarray
    w_theta: jnp.ndarray
    w_zeta: jnp.ndarray
    collision: jnp.ndarray
    spacing: jnp.ndarray
    pin_weight: jnp.ndarray
    pin_sigma: jnp.ndarray
    drive_radial: jnp.ndarray
    drive_epar: jnp.ndarray
    theta_weights: jnp.ndarray
    zeta_weights: jnp.ndarray
    pitch_weights: jnp.ndarray
    surface: FluxSurface
    species_params: SpeciesParams
    delta: jnp.ndarray
    normalization_alpha: jnp.ndarray

    n_x: int
    n_alpha: int
    n_theta: int
    n_zeta: int
    n_periods: int
    stencil: str
    relaxation_stencil: str

    _LEAVES = (
        "x", "x_weights", "alpha", "xi", "theta", "zeta",
        "d_alpha", "d_theta", "d_zeta", "w_alpha", "w_theta", "w_zeta",
        "collision", "spacing", "pin_weight", "pin_sigma",
        "drive_radial", "drive_epar",
        "theta_weights", "zeta_weights", "pitch_weights",
        "surface", "species_params", "delta", "normalization_alpha",
    )  # fmt: skip
    _STATIC = (
        "n_x", "n_alpha", "n_theta", "n_zeta", "n_periods", "stencil", "relaxation_stencil",
    )  # fmt: skip

    def tree_flatten(self):
        return (
            tuple(getattr(self, name) for name in self._LEAVES),
            tuple(getattr(self, name) for name in self._STATIC),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(**dict(zip(cls._LEAVES, children)), **dict(zip(cls._STATIC, aux)))

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """State shape ``(n_x, n_alpha, n_theta, n_zeta)``."""
        return (self.n_x, self.n_alpha, self.n_theta, self.n_zeta)

    @property
    def size(self) -> int:
        """Number of unknowns."""
        return math.prod(self.shape)

    # -- operator --------------------------------------------------------
    def _advect(self, f: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Upwinded ``w df/dcoord`` along state ``axis`` (1 alpha, 2 theta, 3 zeta)."""
        matrices, wind = (
            (self.d_alpha, self.w_alpha),
            (self.d_theta, self.w_theta),
            (self.d_zeta, self.w_zeta),
        )[axis - 1]
        terms = ("ij,xjtz->xitz", "ij,xajz->xaiz", "ij,xatj->xati")[axis - 1]
        return wind * jnp.where(
            wind > 0.0,
            jnp.einsum(terms, matrices[0], f),
            jnp.einsum(terms, matrices[1], f),
        )

    def _collide(self, f: jnp.ndarray) -> jnp.ndarray:
        """Pitch-angle scattering; the end coefficients vanish, so the roll is exact."""
        lower, diag, upper = (band[:, :, None, None] for band in self.collision)
        return diag * f + lower * jnp.roll(f, 1, axis=1) + upper * jnp.roll(f, -1, axis=1)

    def constraint(self, f: jnp.ndarray) -> jnp.ndarray:
        """Flux-surface and pitch average of ``f`` per speed node, ``(X,)``.

        The ``constraintScheme = 2`` constraint row of ``populateMatrix.F90``,
        normalized so a constant state maps to itself.
        """
        return jnp.einsum("atz,xatz->x", self.pin_weight, f)

    def project(self, f: jnp.ndarray) -> jnp.ndarray:
        """Remove the null-space component, giving the constrained solution."""
        return f - self.constraint(f)[:, None, None, None]

    def apply_kinetic(self, f: jnp.ndarray) -> jnp.ndarray:
        """The singular kinetic operator ``K`` (streaming, mirror, ``ExB``, collisions)."""
        return sum(self._advect(f, axis) for axis in (1, 2, 3)) + self._collide(f)

    def apply(self, f: jnp.ndarray) -> jnp.ndarray:
        """Apply the rank-one-shifted operator ``M = K + sigma e m^T``.

        Args:
            f: State of shape :attr:`shape`.

        Returns:
            ``M f``, same shape.
        """
        f = jnp.asarray(f, dtype=jnp.float64)
        if f.shape != self.shape:
            raise ValueError(f"f must have shape {self.shape}, got {f.shape}")
        pin = (self.pin_sigma * self.constraint(f))[:, None, None, None]
        return self.apply_kinetic(f) + pin

    def rhs(self) -> jnp.ndarray:
        """The ``RHSMode = 1`` drive on the collocation grid.

        The gradient drive carries the modal ``4/3 P_0 + 2/3 P_2`` pitch shape,
        which is ``1 + xi^2``; the inductive drive carries ``P_1 = xi``.
        """
        pitch = 1.0 + self.xi * self.xi
        return (
            self.drive_radial[:, None, :, :] * pitch[None, :, None, None]
            + self.drive_epar[:, None, :, :] * self.xi[None, :, None, None]
        )

    # -- relaxation ------------------------------------------------------
    def _wind_bands(self, axis: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """First-order-upwind ``(lower, diag, upper)`` of one advection term.

        On the pitch axis the closure is zero-gradient, so a band pointing off
        the grid acts on the end node itself: it is folded onto the diagonal and
        zeroed, which keeps these bands the *exact* tridiagonal part of the
        assembled operator rather than an approximation of it.
        """
        wind, spacing = (
            (self.w_alpha, self.spacing[0]),
            (self.w_theta, self.spacing[1]),
            (self.w_zeta, self.spacing[2]),
        )[axis - 1]
        forward, backward = _band_weights(self.relaxation_stencil)
        scale = wind / spacing
        lower, diag, upper = (
            scale * jnp.where(wind > 0.0, forward[k], backward[k]) for k in (-1, 0, 1)
        )
        if axis == 1:
            edge = jnp.arange(self.n_alpha)[None, :, None, None]
            first, last = edge == 0, edge == self.n_alpha - 1
            diag = diag + jnp.where(first, lower, 0.0) + jnp.where(last, upper, 0.0)
            lower = jnp.where(first, 0.0, lower)
            upper = jnp.where(last, 0.0, upper)
        return lower, diag, upper

    def diagonal(self) -> jnp.ndarray:
        """Main diagonal of the kinetic operator ``K``, ``(X, A, T, Z)``.

        The rank-one shift is deliberately excluded: the relaxation and the
        coarse levels run on the unshifted physics (see :func:`_preconditioner`).
        """
        return self.collision[1][:, :, None, None] + sum(
            self._wind_bands(axis)[1] for axis in (1, 2, 3)
        )

    def line_bands(self, axis: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """``(lower, diag, upper)`` of the line operator along state ``axis``.

        ``diag`` is the *full* operator diagonal, so an exact solve of these
        bands is one line block of a block-Jacobi relaxation.  With
        ``stencil = relaxation_stencil = "up1"`` the bands are the exact line
        blocks; with a widened ``stencil`` they are its first-order surrogate.
        """
        lower, _, upper = self._wind_bands(axis)
        if axis == 1:
            lower = lower + self.collision[0][:, :, None, None]
            upper = upper + self.collision[2][:, :, None, None]
        zeros = jnp.zeros(self.shape)
        return lower + zeros, self.diagonal(), upper + zeros

    # -- diagnostics -----------------------------------------------------
    def legendre_moments(self, f: jnp.ndarray, n_xi: int) -> jnp.ndarray:
        """Project onto Legendre pitch moments, ``(X, n_xi, T, Z)``.

        ``f_l = (2l+1)/2 int P_l(xi) f dxi`` evaluated with the grid's own
        midpoint pitch quadrature, so the result is what dkx's Legendre
        discretization would carry for the same distribution.
        """
        xi = np.asarray(self.xi, dtype=np.float64)
        basis = np.polynomial.legendre.legvander(xi, int(n_xi) - 1).T  # (L, A)
        factor = (2.0 * np.arange(int(n_xi)) + 1.0) / 2.0
        weights = jnp.asarray(basis * factor[:, None]) * self.pitch_weights[None, :]
        return jnp.einsum("la,xatz->xltz", weights, f)

    def flux_moments(self, f: jnp.ndarray, *, n_xi: int = 4):
        """Radial fluxes and ``FSABFlow`` of a collocation state.

        Embeds the state in dkx's Legendre layout and hands it to
        :func:`dkx.moments.vm_flux_moments`, so the outputs are the *same*
        functionals, with the same normalization, that the modal path reports --
        which is what makes a cross-discretization comparison meaningful.  Only
        ``l <= 2`` enters those integrals; ``n_xi`` merely sizes the layout.

        Args:
            f: State of shape :attr:`shape`.  Pass the projected solution.
            n_xi: Number of Legendre moments to carry into the layout.

        Returns:
            A :class:`dkx.moments.VmFluxMoments` for the single species.
        """
        moments = self.legendre_moments(f, n_xi)[None]  # (S=1, X, L, T, Z)
        layout = StateLayout(
            n_species=1, n_x=self.n_x, n_xi=int(n_xi), n_theta=self.n_theta, n_zeta=self.n_zeta
        )
        vgrid = VelocityGrid(
            x=self.x,
            x_weights=self.x_weights,
            n_xi_for_x=jnp.full((self.n_x,), int(n_xi), dtype=jnp.int32),
        )
        return vm_flux_moments(
            layout,
            vgrid,
            self.surface,
            self.species_params,
            moments.reshape(-1),
            delta=self.delta,
            alpha=self.normalization_alpha,
        )


def collocation_operator(
    *,
    geometry: FluxSurfaceGeometry,
    theta: np.ndarray,
    zeta: np.ndarray,
    n_alpha: int,
    x: np.ndarray,
    x_weights: np.ndarray,
    nu_d_hat: np.ndarray,
    nu_n: float,
    z_s: float,
    m_hat: float,
    t_hat: float,
    n_hat: float,
    dn_hat_dpsi_hat: float,
    dt_hat_dpsi_hat: float,
    dphi_hat_dpsi_hat: float,
    normalization_alpha: float,
    delta: float,
    e_parallel_hat: float = 0.0,
    use_dkes_exb: bool = False,
    with_exb: bool = True,
    stencil: str = "up1",
    relaxation_stencil: str = "up1",
) -> CollocationOperator:
    """Assemble a :class:`CollocationOperator` from geometry and one species.

    Every argument follows the SFINCS v3 "Hat" normalization and the naming of
    :class:`dkx.drift_kinetic.KineticOperator`;
    :func:`collocation_operator_from_namelist` is the convenient way to obtain
    them all consistently.

    Args:
        geometry: Flux-surface geometry evaluated on ``theta`` x ``zeta``.
        theta: Uniform poloidal grid over ``[0, 2 pi)``.
        zeta: Uniform toroidal grid over ``[0, 2 pi / Nperiods)``.
        n_alpha: Number of pitch-angle nodes.
        x: Speed nodes ``v / v_th``.
        x_weights: Speed quadrature weights (they include ``exp(-x^2)``).
        nu_d_hat: Deflection frequency ``nuDHat(x)`` of the species.
        nu_n: Namelist ``nu_n``.
        z_s: Charge number.
        m_hat: Mass over the reference mass.
        t_hat: Temperature over the reference temperature.
        n_hat: Density over the reference density.
        dn_hat_dpsi_hat: ``dnHat/dpsiHat``.
        dt_hat_dpsi_hat: ``dTHat/dpsiHat``.
        dphi_hat_dpsi_hat: ``dPhiHat/dpsiHat``, the radial electric field.
        normalization_alpha: Namelist ``alpha``.
        delta: Namelist ``Delta``.
        e_parallel_hat: Inductive parallel electric field.
        use_dkes_exb: Replace ``BHat^2`` by ``<BHat^2>`` in the ``ExB`` drift.
        with_exb: Include the ``ExB`` drift.
        stencil: Upwind stencil of the operator.
        relaxation_stencil: Upwind stencil of the relaxation bands.

    Returns:
        The assembled operator, with ``sigma`` already set to the mean diagonal.
    """
    theta = np.asarray(theta, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n_theta, n_zeta = theta.size, zeta.size
    n_periods = int(geometry.n_periods)
    n_alpha = int(n_alpha)

    alpha = np.pi * (2.0 * np.arange(n_alpha) + 1.0) / (2.0 * n_alpha)
    xi = np.cos(alpha)
    h_alpha = math.pi / n_alpha
    h_theta = 2.0 * math.pi / n_theta
    h_zeta = 2.0 * math.pi / (n_periods * n_zeta)

    def geo(name: str) -> np.ndarray:
        return np.asarray(getattr(geometry, name), dtype=np.float64)

    b_hat, d_hat = geo("b_hat"), geo("d_hat")
    db_dtheta, db_dzeta = geo("db_hat_dtheta"), geo("db_hat_dzeta")
    b_sub_theta, b_sub_zeta = geo("b_hat_sub_theta"), geo("b_hat_sub_zeta")
    b_sup_theta, b_sup_zeta = geo("b_hat_sup_theta"), geo("b_hat_sup_zeta")

    theta_weights = np.full(n_theta, 2.0 * math.pi / n_theta)
    zeta_weights = np.full(n_zeta, 2.0 * math.pi / n_zeta)
    weight_2d = theta_weights[:, None] * zeta_weights[None, :]
    vprime_hat = float(np.sum(weight_2d / d_hat))
    fsab_hat2 = float(np.sum(weight_2d * b_hat**2 / d_hat) / vprime_hat)

    # -- trajectories ----------------------------------------------------
    speed = math.sqrt(t_hat / m_hat) * x  # a = x sqrt(THat/mHat)
    b_grad_ln_b = (b_sup_theta * db_dtheta + b_sup_zeta * db_dzeta) / b_hat**2
    if with_exb:
        denom = fsab_hat2 if use_dkes_exb else b_hat**2
        factor = normalization_alpha * delta * 0.5 * dphi_hat_dpsi_hat
        exb_theta, exb_zeta = factor * d_hat * b_sub_zeta / denom, -factor * d_hat * b_sub_theta / denom
    else:
        exb_theta = exb_zeta = np.zeros_like(b_hat)

    shape = (x.size, n_alpha, n_theta, n_zeta)
    # dalpha/dt follows from dxi/dt and xi = cos(alpha): the (1-xi^2)/2 mirror
    # factor becomes sin(alpha)/2 once the chain rule's 1/sin(alpha) is applied.
    w_alpha = np.broadcast_to(
        speed[:, None, None, None]
        * (0.5 * np.sin(alpha))[None, :, None, None]
        * b_grad_ln_b[None, None, :, :],
        shape,
    )
    w_theta = np.broadcast_to(
        speed[:, None, None, None] * xi[None, :, None, None] * (b_sup_theta / b_hat)[None, None]
        + exb_theta[None, None],
        shape,
    )
    w_zeta = np.broadcast_to(
        speed[:, None, None, None] * xi[None, :, None, None] * (b_sup_zeta / b_hat)[None, None]
        + exb_zeta[None, None],
        shape,
    )

    # -- collisions (conservative flux form; see the module docstring) ----
    nu = float(nu_n) * np.asarray(nu_d_hat, dtype=np.float64)
    s_up = np.sin(alpha + 0.5 * h_alpha)
    s_down = np.sin(alpha - 0.5 * h_alpha)
    s_up[-1] = s_down[0] = 0.0  # sin vanishes at both ends: no-flux, exactly
    scale = 0.5 * nu[:, None] / (np.sin(alpha)[None, :] * h_alpha * h_alpha)
    collision = np.stack([-scale * s_down, scale * (s_up + s_down), -scale * s_up])

    # -- drives (evaluateResidual.F90, RHSMode = 1) ------------------------
    x2 = x * x
    expx2 = np.exp(-x2)
    geom2 = (b_sub_zeta * db_dtheta - b_sub_theta * db_dzeta) * d_hat / b_hat**3
    x_part = x2 * expx2 * (
        dn_hat_dpsi_hat / n_hat
        + normalization_alpha * z_s / t_hat * dphi_hat_dpsi_hat
        + (x2 - 1.5) * dt_hat_dpsi_hat / t_hat
    )
    pref = (
        delta * n_hat * m_hat * math.sqrt(m_hat)
        / (2.0 * math.pi * math.sqrt(math.pi) * z_s * math.sqrt(t_hat))
    )
    factor_e = (
        normalization_alpha * z_s * x * expx2 * e_parallel_hat * n_hat * m_hat
        / (math.pi * math.sqrt(math.pi) * t_hat * t_hat * fsab_hat2)
    )

    # -- null-space closure ------------------------------------------------
    pitch_weights = h_alpha * np.sin(alpha)
    pin_weight = pitch_weights[:, None, None] * (weight_2d / d_hat)[None]
    pin_weight /= np.sum(pin_weight)

    def diffs(n: int, spacing: float, periodic: bool) -> jnp.ndarray:
        return jnp.asarray(
            np.stack(_diff_matrices(n=n, spacing=spacing, stencil=stencil, periodic=periodic))
        )

    scalars = jnp.asarray([float(z_s)]), jnp.asarray([float(m_hat)])
    operator = CollocationOperator(
        x=jnp.asarray(x),
        x_weights=jnp.asarray(np.asarray(x_weights, dtype=np.float64)),
        alpha=jnp.asarray(alpha),
        xi=jnp.asarray(xi),
        theta=jnp.asarray(theta),
        zeta=jnp.asarray(zeta),
        d_alpha=diffs(n_alpha, h_alpha, False),
        d_theta=diffs(n_theta, h_theta, True),
        d_zeta=diffs(n_zeta, h_zeta, True),
        w_alpha=jnp.asarray(w_alpha),
        w_theta=jnp.asarray(w_theta),
        w_zeta=jnp.asarray(w_zeta),
        collision=jnp.asarray(collision),
        spacing=jnp.asarray([h_alpha, h_theta, h_zeta]),
        pin_weight=jnp.asarray(pin_weight),
        pin_sigma=jnp.zeros((x.size,)),
        drive_radial=jnp.asarray(pref * x_part[:, None, None] * geom2[None]),
        drive_epar=jnp.asarray(factor_e[:, None, None] * b_hat[None]),
        theta_weights=jnp.asarray(theta_weights),
        zeta_weights=jnp.asarray(zeta_weights),
        pitch_weights=jnp.asarray(pitch_weights),
        surface=FluxSurface(
            theta_weights=jnp.asarray(theta_weights),
            zeta_weights=jnp.asarray(zeta_weights),
            b_hat=jnp.asarray(b_hat),
            d_hat=jnp.asarray(d_hat),
            db_hat_dtheta=jnp.asarray(db_dtheta),
            db_hat_dzeta=jnp.asarray(db_dzeta),
            b_hat_sub_theta=jnp.asarray(b_sub_theta),
            b_hat_sub_zeta=jnp.asarray(b_sub_zeta),
            fsab_hat2=jnp.asarray(fsab_hat2),
        ),
        species_params=SpeciesParams(
            z_s=scalars[0],
            m_hat=scalars[1],
            t_hat=jnp.asarray([float(t_hat)]),
            n_hat=jnp.asarray([float(n_hat)]),
        ),
        delta=jnp.asarray(float(delta)),
        normalization_alpha=jnp.asarray(float(normalization_alpha)),
        n_x=int(x.size),
        n_alpha=n_alpha,
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
        n_periods=n_periods,
        stencil=str(stencil),
        relaxation_stencil=str(relaxation_stencil),
    )
    # sigma is the mean diagonal of K: large enough that the shifted operator is
    # not near-singular, small enough that it does not dominate the physics.
    return replace(operator, pin_sigma=jnp.mean(operator.diagonal(), axis=(1, 2, 3)))


def collocation_operator_from_namelist(
    nml: Any, options: CollocationOptions, *, grid: tuple[int, int, int] | None = None
) -> CollocationOperator:
    """Build the collocation operator for a SFINCS input namelist.

    Angular resolution comes from ``options`` (or ``grid``), not from
    ``resolutionParameters``: the collocation grid wants *even* lengths, which
    ``Ntheta``/``Nzeta`` are forbidden to be on the modal path.  ``Nx`` and the
    speed-grid options are honoured, ``Nxi`` is ignored, and geometry is
    re-evaluated on the requested angular grid -- which is exactly what makes
    the rediscretized coarse levels of :func:`solve_collocation` cheap.

    Args:
        nml: Parsed namelist (:func:`dkx.inputs.parse_sfincs_input_text`).
        options: Discretization settings; supplies the default grid, the
            stencils and the species index.
        grid: Override ``(n_alpha, n_theta, n_zeta)``, used to rediscretize the
            same physics on a coarse level.

    Returns:
        The assembled operator.

    Raises:
        NotImplementedError: If the namelist selects physics outside this
            backend -- anything but ``collisionOperator = 1`` at ``RHSMode = 1``
            without ``Phi1``, magnetic drifts, or the ``E_r`` ``xDot``/``xiDot``
            terms.
    """
    build = kinetic_operator_build_from_namelist(nml)
    op = build.operator
    if op.pas is None or op.fp is not None or op.sugama is not None:
        raise NotImplementedError(
            "the collocation backend implements collisionOperator=1 (pitch-angle scattering) only"
        )
    for flag in ("include_phi1", "with_magnetic_drifts", "with_er_xidot", "with_er_xdot"):
        if getattr(op, flag, False):
            raise NotImplementedError(f"the collocation backend does not implement {flag}")
    if int(op.rhs_mode) != 1:
        raise NotImplementedError("the collocation backend implements RHSMode=1 only")

    n_alpha, n_theta, n_zeta = grid if grid is not None else options.grid
    n_periods = int(build.grids.n_periods)
    theta = 2.0 * math.pi * np.arange(n_theta) / n_theta
    zeta = 2.0 * math.pi * np.arange(n_zeta) / (n_periods * n_zeta)
    # _geometry_and_radial reads only ``grids.theta``/``grids.zeta``; passing our
    # own grid is what re-evaluates |B| and its metrics on a coarse level.
    geometry, _ = _geometry_and_radial(
        nml=nml, grids=SimpleNamespace(theta=jnp.asarray(theta), zeta=jnp.asarray(zeta))
    )
    s = int(options.species)
    scalar = lambda name: float(np.asarray(getattr(op, name)))  # noqa: E731
    per_species = lambda name: float(np.asarray(getattr(op, name))[s])  # noqa: E731
    return collocation_operator(
        geometry=geometry,
        theta=theta,
        zeta=zeta,
        n_alpha=n_alpha,
        x=np.asarray(op.x),
        x_weights=np.asarray(op.x_weights),
        nu_d_hat=np.asarray(op.pas.nu_d_hat)[s],
        nu_n=float(op.pas.nu_n),
        z_s=per_species("z_s"),
        m_hat=per_species("m_hat"),
        t_hat=per_species("t_hat"),
        n_hat=per_species("n_hat"),
        dn_hat_dpsi_hat=per_species("dn_hat_dpsi_hat"),
        dt_hat_dpsi_hat=per_species("dt_hat_dpsi_hat"),
        dphi_hat_dpsi_hat=scalar("dphi_hat_dpsi_hat"),
        normalization_alpha=scalar("alpha"),
        delta=scalar("delta"),
        e_parallel_hat=scalar("e_parallel_hat") + per_species("e_parallel_hat_spec"),
        use_dkes_exb=bool(op.use_dkes_exb),
        with_exb=bool(op.with_exb),
        stencil=options.stencil,
        relaxation_stencil=options.relaxation_stencil,
    )


def _smoother(op: CollocationOperator, options: CollocationOptions):
    """Relaxation for one multigrid level.

    ``"line"`` (the default) inverts each grid direction in turn, exactly, with
    the full operator diagonal: an alternating line block-Jacobi relaxation over
    the three coarsened axes, which is the standard response to an advection not
    aligned with the mesh (Trottenberg et al., sections 5.1 and 7.4).  Undamped
    (``omega = 1``) it contracts at roughly ``0.5`` per cycle on *every* speed
    node of the W7-X deck -- from the collision-dominated lowest node to the
    essentially collisionless highest, seven decades of mesh Peclet number
    apart -- provided the null-space shift is kept out of the cycle
    (:func:`_preconditioner`).  With the shift inside, no weight works for all
    of them at once.

    ``"plane"`` is the stronger and costlier alternative: it inverts the whole
    ``(theta, zeta)`` plane at each pitch node, so the entire streaming and
    ``ExB`` advection -- which lives in that plane, along field lines aligned
    with neither angular axis -- is solved exactly and only the mirror coupling
    in ``alpha`` is left to the pitch lines.  Plane relaxation is the classical
    answer when an operator is strongly coupled along two axes at once
    (Trottenberg et al., section 5.2); measured here it matches the line sweep
    rather than beating it, at ``O(n_theta n_zeta^2)`` factors per plane instead
    of ``O(n_theta n_zeta)``.

    ``"upwind"`` replaces the exact line solves by sweeps ordered along the wind
    (Brandt & Yavneh 1993), which costs the same and measures the same here.
    """
    from solvax.smoothers import (
        alternating_smoother,
        plane_smoother,
        tridiagonal_smoother,
        upwind_smoother,
    )

    weight = float(options.omega)
    if options.smoother == "plane":
        lower, diagonal, upper = op.line_bands(1)
        theta_lower, _, theta_upper = op.line_bands(2)
        zeta_lower, _, zeta_upper = op.line_bands(3)
        return alternating_smoother([
            plane_smoother(
                diagonal,
                (theta_lower, theta_upper),
                (zeta_lower, zeta_upper),
                axes=(2, 3),
                periodic=(True, True),
                omega=weight,
            ),
            tridiagonal_smoother(lower, diagonal, upper, axis=1, periodic=False, omega=weight),
        ])

    winds = (op.w_alpha, op.w_theta, op.w_zeta)
    sweeps = []
    for axis in (1, 2, 3):
        bands = op.line_bands(axis)
        common = dict(axis=axis, periodic=axis != 1, omega=weight)
        if options.smoother == "line":
            sweeps.append(tridiagonal_smoother(*bands, **common))
        else:
            sweeps.append(upwind_smoother(winds[axis - 1], *bands, **common))
    return alternating_smoother(sweeps)


def _preconditioner(
    op: CollocationOperator, options: CollocationOptions, rediscretize
) -> tuple[Any, tuple[tuple[int, ...], ...]]:
    """Semicoarsened multigrid over ``(alpha, theta, zeta)``, ``x`` kept fine.

    Coarse operators are **rediscretized** -- the same physics, geometry
    included, rebuilt on the coarse grid -- rather than Galerkin products, which
    keeps every level in the banded form the smoothers need (Trottenberg et al.,
    ch. 2 and 5).  Transfers are the separable per-axis linear interpolation /
    full weighting of :mod:`solvax.transfer`, periodic on the angles and
    cell-centered (``"reflective"``) on the half-index pitch grid.  The recursion
    bottoms out on an exact dense solve, one factorization per speed node
    because pitch-angle scattering leaves the operator block diagonal in ``x``.

    The hierarchy runs on the **unshifted, singular** ``K``, not on the ``M`` the
    outer Krylov method sees.  Putting the rank-one shift inside the cycle is
    what destroys it, and measurably so: the shift is a global dense operator
    that no local relaxation can see, so its entire correction must come from
    the coarse grids -- yet its natural strength is the mean diagonal, which
    rediscretizes with the mesh (as ``h^-2`` wherever collisions dominate).
    Every level then corrects the null-space mode by a different factor and the
    cycle amplifies it; measured on W7-X, the cycle diverges by ten orders of
    magnitude per application on the collision-dominated speed nodes, and
    freezing the shift instead makes it worse, because a fine-grid-sized shift
    swamps a coarse operator.

    The cure is the standard treatment of a singular system (Trottenberg et al.,
    section 5.6): cycle on the singular physics, project the residual and the
    correction onto the constrained subspace ``m^T x = 0``, and let the coarsest
    level supply the pseudo-inverse (its exact pinned solve, projected).  The
    null direction is then handled exactly and separately -- ``M e = sigma e``,
    so ``M^{-1}`` acts there as ``1/sigma``.  With that split the cycle needs no
    damping at all.

    Args:
        op: the finest-level operator.
        options: cycle, smoother and coarsening settings.
        rediscretize: ``(options, (n_alpha, n_theta, n_zeta)) -> operator``.
            It is called with the *coarse* options, whose ``stencil`` is
            ``relaxation_stencil``: that is the other half of double
            discretization, and a widened stencil is in any case wider than the
            coarsest grids the hierarchy reaches.

    Returns:
        ``(precond, shapes)``: the ``r -> M^{-1} r`` callable and the grid shapes
        of the hierarchy, finest first.
    """
    from solvax.precond import dense_coarse_solve, multigrid, semicoarsening_hierarchy

    coarse_options = replace(options, stencil=options.relaxation_stencil)

    def level_at(shape) -> CollocationOperator:
        return op if tuple(shape) == op.shape else rediscretize(coarse_options, shape[1:])

    hierarchy = semicoarsening_hierarchy(
        op.shape,
        (False, True, True, True),
        lambda shape: ((level := level_at(shape)).apply_kinetic, _smoother(level, options)),
        levels=int(options.levels),
        boundary=("periodic", "reflective", "periodic", "periodic"),
        min_size=int(options.min_coarse),
    )
    coarsest = hierarchy.shapes[-1]
    coarse_op = level_at(coarsest)
    exact = dense_coarse_solve(coarse_op.apply, coarsest, batch_dims=1)
    cycle = multigrid(
        hierarchy.levels,
        lambda b: coarse_op.project(exact(b)),
        cycle=options.cycle,
        pre_smooth=int(options.pre_smooth),
        post_smooth=int(options.post_smooth),
    )

    def precond(r: jnp.ndarray) -> jnp.ndarray:
        null = op.constraint(r)[:, None, None, None] / op.pin_sigma[:, None, None, None]
        return op.project(cycle(op.project(r))) + null

    return precond, hierarchy.shapes


@dataclass(frozen=True)
class CollocationSolution:
    """Result of :func:`solve_collocation`.

    Attributes:
        f: Constrained solution on the collocation grid, ``(X, A, T, Z)``.
        operator: The finest-level operator that produced it.
        residual: True relative residual ``||b - M f|| / ||b||``.
        iterations: Outer Krylov iterations.
        converged: Whether the Krylov method met its own tolerance.
        hierarchy: Multigrid grid shapes, finest first; empty when
            unpreconditioned.
    """

    f: jnp.ndarray
    operator: CollocationOperator
    residual: float
    iterations: int
    converged: bool
    hierarchy: tuple[tuple[int, ...], ...]

    def flux_moments(self, *, n_xi: int = 4):
        """Radial fluxes and ``FSABFlow``; see :meth:`CollocationOperator.flux_moments`."""
        return self.operator.flux_moments(self.f, n_xi=n_xi)


def solve_collocation(
    nml: Any,
    options: CollocationOptions | None = None,
    solver: SolverOptions | None = None,
) -> CollocationSolution:
    """Solve the drift-kinetic equation on the pitch-collocation grid.

    The outer method is GCROT with harmonic (GCRO-DR) recycling, preconditioned
    by the semicoarsened multigrid cycle of :func:`_preconditioner`; the
    solution is then projected onto the ``constraintScheme = 2`` constraint.

    Args:
        nml: Parsed namelist (:func:`dkx.inputs.parse_sfincs_input_text`).
        options: Discretization and multigrid settings; defaults to
            :class:`CollocationOptions`.
        solver: Outer Krylov settings; ``tol``, ``restart``, ``recycle_dim`` and
            ``max_restarts`` are used.  Defaults to :class:`dkx.api.SolverOptions`.

    Returns:
        A :class:`CollocationSolution`.
    """
    from solvax.krylov import gcrot

    options = options or CollocationOptions()
    solver = solver or SolverOptions()
    op = collocation_operator_from_namelist(nml, options)

    precond, shapes = (None, ())
    if options.preconditioner == "multigrid":
        precond, shapes = _preconditioner(
            op,
            options,
            lambda coarse, grid: collocation_operator_from_namelist(nml, coarse, grid=tuple(grid)),
        )

    b = op.rhs()
    solution = gcrot(
        jax.jit(op.apply),
        b,
        precond=None if precond is None else jax.jit(precond),
        m=int(solver.restart),
        k=int(solver.recycle_dim),
        rtol=float(solver.tol),
        atol=float(solver.atol),
        max_restarts=int(solver.max_restarts),
        recycle_strategy="harmonic",
    )
    residual = float(jnp.linalg.norm(b - op.apply(solution.x)) / jnp.linalg.norm(b))
    return CollocationSolution(
        f=op.project(solution.x),
        operator=op,
        residual=residual,
        iterations=int(solution.iterations),
        converged=bool(solution.converged),
        hierarchy=tuple(tuple(int(n) for n in shape) for shape in shapes),
    )
