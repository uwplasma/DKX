"""Variational (entropy-production) bounds on the monoenergetic D11 coefficient.

The RHSMode=3 monoenergetic drift-kinetic system has the classic variational
structure of the drift-kinetic equation with pitch-angle scattering: writing
the operator as ``M = V + P`` where ``P`` (collisions) is symmetric positive
semidefinite and ``V`` (parallel streaming + mirror + ExB) is antisymmetric
under the entropy inner product, the radial-gradient drive is even in the
pitch coordinate and the diffusion coefficient is the entropy production of
the solution, ``D = <s, f>_W = <f, P f>_W``.  Two quadratic functionals then
bound ``D`` from below and above for *any* trial pair (g even, h odd in
Legendre parity):

- lower:  ``Phi-[g] = 2<s,g> - <g,P g> - <(V g)_odd, P^-1 (V g)_odd>``
- upper:  ``Phi+[h] = <h,P h> + <(V h - s)_even, P^-1 (V h - s)_even>``

Both are tight at the exact solution.  Evaluated at the even/odd parts of the
*discrete* solution, the residual equations collapse most terms and the two
functionals sit symmetrically around the discrete ``D11``: their deviation is
``+-<(V + V^T_W) h, g>_W``, i.e. exactly the antisymmetry defect of the
discrete streaming operator under the entropy weight.  The pair therefore
brackets the computed coefficient to solver-residual precision, and the
relative gap is an a-posteriori certificate of how well the discretization
preserves the continuum entropy-production structure (it vanishes under
theta/zeta/xi refinement and at high collisionality, where ``f`` is
collision-dominated).

The strict bound property of the continuum functionals requires a purely
parity-flipping ``V``, which holds for the monoenergetic trajectory terms at zero
radial electric field; with ``EStar != 0`` the (parity-preserving,
antisymmetric) ExB term contributes to the gap as well and the certificate
remains a consistency diagnostic.

Primary literature: S.P. Hirshman, K.C. Shaing, W.I. van Rij, C.O. Beasley,
and E.C. Crume, Phys. Fluids 29, 2951 (1986) (the variational principle and
the bounding functionals); W.I. van Rij and S.P. Hirshman, Phys. Fluids B 1,
563 (1989) (upper/lower estimates of the monoenergetic coefficients as
convergence certificates).

Everything here uses only canonical operator applies
(:meth:`~sfincs_jax.drift_kinetic.KineticOperator.apply_f` and the
pitch-angle-scattering apply) plus quadrature/geometry arrays already stored
on the operator; no new physics is assembled.  All functions are pure and
jit-friendly.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from .collisions import apply_pitch_angle_scattering_v3  # noqa: E402
from .moments import (  # noqa: E402
    FluxSurface,
    SpeciesParams,
    StateLayout,
    VelocityGrid,
    transport_matrix_from_state_vectors,
    vprime_hat,
)

__all__ = ["MonoenergeticD11Bounds", "d11_bounds_supported", "monoenergetic_d11_bounds"]


class MonoenergeticD11Bounds(NamedTuple):
    """Variational upper/lower bounds bracketing the RHSMode=3 ``transportMatrix[0][0]``.

    Attributes:
        d11: the coefficient reconstructed from the entropy-production form
            ``C <s, f>_W`` (equals ``transportMatrix[0][0]`` to roundoff).
        lower: certified lower bound (the more negative functional value).
        upper: certified upper bound.
        gap: relative convergence certificate ``|upper - lower| / |d11|``.
    """

    d11: jnp.ndarray
    lower: jnp.ndarray
    upper: jnp.ndarray
    gap: jnp.ndarray


def d11_bounds_supported(op: Any) -> bool:
    """True when :func:`monoenergetic_d11_bounds` applies to this operator."""
    return (
        op.rhs_mode == 3
        and op.pas is not None
        and op.fp is None
        and op.fp_phi1 is None
        and not op.include_phi1
        and op.external_phi1_hat is None
        and not op.with_magnetic_drifts
        and not op.with_er_xidot
        and not op.with_er_xdot
        and not op.point_at_x0
    )


def _check_supported(op: Any) -> None:
    if op.rhs_mode != 3:
        raise ValueError("monoenergetic_d11_bounds requires an RHSMode=3 operator.")
    if op.pas is None or op.fp is not None or op.fp_phi1 is not None:
        raise ValueError("monoenergetic_d11_bounds requires pitch-angle-scattering collisions (collisionOperator=1).")
    if op.include_phi1 or op.external_phi1_hat is not None:
        raise ValueError("monoenergetic_d11_bounds does not support Phi1 operators.")
    if op.with_magnetic_drifts or op.with_er_xidot or op.with_er_xdot:
        raise ValueError(
            "monoenergetic_d11_bounds requires monoenergetic trajectories "
            "(no tangential magnetic drifts, no Er xiDot/xDot terms)."
        )
    if op.point_at_x0:
        raise ValueError("monoenergetic_d11_bounds does not support a speed-grid point at x=0.")


def _entropy_weight(op: Any) -> jnp.ndarray:
    """Entropy inner-product weight ``W(s,x,l,theta,zeta)``.

    ``2/(2l+1)`` is the Legendre-mode norm, ``w_x x^2 e^{x^2}`` the speed
    measure over the Maxwellian, and ``w_theta w_zeta / DHat`` the
    flux-surface volume element.  Under this weight the PAS operator is a
    positive diagonal and the streaming + mirror (+ incompressible ExB) terms are
    antisymmetric up to discretization error.
    """
    ell = jnp.arange(op.n_xi, dtype=jnp.float64)
    w_l = 2.0 / (2.0 * ell + 1.0)
    w_x = op.x_weights * op.x**2 * jnp.exp(op.x**2)
    w_tz = (op.theta_weights[:, None] * op.zeta_weights[None, :]) / op.d_hat
    mask = (jnp.arange(op.n_xi)[None, :] < op.n_xi_for_x[:, None]).astype(jnp.float64)  # (X,L)
    return (
        w_x[None, :, None, None, None]
        * (w_l[None, :] * mask)[None, :, :, None, None]
        * w_tz[None, None, None, :, :]
        * jnp.ones((op.n_species, 1, 1, 1, 1), dtype=jnp.float64)
    )


def _d11_prefactor(
    op: Any,
    *,
    g_hat: jnp.ndarray,
    i_hat: jnp.ndarray,
    iota: jnp.ndarray,
    b0_over_bbar: jnp.ndarray,
) -> jnp.ndarray:
    """Constant ``C`` with ``transportMatrix[0][0] = C <s_1, f_1>_W + offset``.

    Combines the ``diagnostics.F90`` RHSMode=3 (1,1) entry prefactor with the
    ratio of the vm particle-flux moment kernel to the whichRHS=1 gradient
    drive; the two kernels are proportional through the entropy weight, which
    is what makes the D11 entry an entropy production.
    """
    t_hat = op.t_hat[0]
    m_hat = op.m_hat[0]
    n_hat = op.n_hat[0]
    z = op.z_s[0]
    vp = vprime_hat(FluxSurface.from_operator(op))
    g_plus = g_hat + iota * i_hat
    return (
        -8.0
        * jnp.pi**2
        * jnp.sqrt(jnp.pi)
        * z**2
        * b0_over_bbar
        * g_plus
        * t_hat**1.5
        / (op.delta**2 * g_hat**2 * vp * n_hat * m_hat**3.5)
    )


def monoenergetic_d11_bounds(
    op: Any,
    state_vector: jnp.ndarray,
    *,
    g_hat: jnp.ndarray | float,
    i_hat: jnp.ndarray | float,
    iota: jnp.ndarray | float,
    b0_over_bbar: jnp.ndarray | float,
) -> MonoenergeticD11Bounds:
    """Evaluate the variational D11 bounds from a converged whichRHS=1 state.

    Args:
        op: the RHSMode=3 :class:`~sfincs_jax.drift_kinetic.KineticOperator`
            (PAS collisions, monoenergetic trajectories).
        state_vector: solved whichRHS=1 state, shape ``(total_size,)``.
        g_hat / i_hat / iota / b0_over_bbar: flux functions used by the
            ``diagnostics.F90`` transport-matrix normalization (placeholder
            values are substituted exactly as in
            :func:`sfincs_jax.moments.transport_matrix_from_flux_arrays`).

    Returns:
        A :class:`MonoenergeticD11Bounds`; ``lower <= transportMatrix[0][0]
        <= upper`` holds to solver-residual precision and ``gap`` is the
        relative discretization certificate.
    """
    _check_supported(op)
    state_vector = jnp.asarray(state_vector, dtype=jnp.float64)
    if state_vector.shape != (op.total_size,):
        raise ValueError(f"state_vector must have shape {(op.total_size,)}, got {state_vector.shape}")

    weight = _entropy_weight(op)
    f5 = state_vector[: op.f_size].reshape(op.f_shape)
    extras = state_vector[op.f_size :]

    # Effective drive of the constrained problem: the whichRHS=1 RHS minus the
    # bordered particle-source injection (populateMatrix.F90 source columns).
    s5 = op.rhs(1)[: op.f_size].reshape(op.f_shape)
    if op.constraint_scheme in (1, 3, 4):
        src = extras.reshape((op.n_species, 2))
        xpart1, xpart2 = op._source_basis(op.constraint_scheme)
        s5 = s5.at[:, :, 0, :, :].add(
            -(xpart1[None, :, None, None] * src[:, 0, None, None, None])
            - (xpart2[None, :, None, None] * src[:, 1, None, None, None])
        )
    elif op.constraint_scheme == 2:
        src = extras.reshape((op.n_species, op.n_x))
        s5 = s5.at[:, :, 0, :, :].add(-src[:, :, None, None])

    # Symmetric-positive collision diagonal (one PAS apply on a ones field)
    # and its pseudo-inverse; the L=0 nullspace column is annihilated, which
    # is exact for the constrained problem (its L=0 balance is enforced by
    # the constraint rows, not by collisions).
    coef = apply_pitch_angle_scattering_v3(op.pas, jnp.ones(op.f_shape, dtype=jnp.float64))
    pinv = jnp.where(coef > 0.0, 1.0 / jnp.where(coef > 0.0, coef, 1.0), 0.0)

    ell = jnp.arange(op.n_xi)
    even5 = jnp.where((ell % 2 == 0)[None, None, :, None, None], 1.0, 0.0)
    g5 = f5 * even5
    h5 = f5 * (1.0 - even5)

    def apply_v(v5: jnp.ndarray) -> jnp.ndarray:
        """Antisymmetric part carrier: full kinetic f-block apply minus collisions."""
        return op.apply_f(v5) - coef * v5

    def wdot(a5: jnp.ndarray, b5: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(weight * a5 * b5)

    v_g = apply_v(g5)
    v_h = apply_v(h5)

    # Lower functional (Hirshman et al. 1986): maximized by the exact even part.
    vg_odd = v_g * (1.0 - even5)
    phi_minus = 2.0 * wdot(s5, g5) - wdot(g5, coef * g5) - wdot(vg_odd, pinv * vg_odd)

    # Upper functional: minimized by the exact odd part.
    r_even = (v_h - s5) * even5
    phi_plus = wdot(h5, coef * h5) + wdot(r_even, pinv * r_even)

    c = _d11_prefactor(
        op,
        g_hat=jnp.asarray(g_hat, dtype=jnp.float64),
        i_hat=jnp.asarray(i_hat, dtype=jnp.float64),
        iota=jnp.asarray(iota, dtype=jnp.float64),
        b0_over_bbar=jnp.asarray(b0_over_bbar, dtype=jnp.float64),
    )

    # Maxwellian-f0 offset of the diagnostics particle-flux moment (state
    # independent; zero to quadrature accuracy but included for exactness).
    layout = StateLayout(
        n_species=op.n_species, n_x=op.n_x, n_xi=op.n_xi, n_theta=op.n_theta,
        n_zeta=op.n_zeta, include_phi1=bool(op.include_phi1), constraint_scheme=op.constraint_scheme,
    )  # fmt: skip
    vgrid = VelocityGrid(
        x=jnp.asarray(op.x, dtype=jnp.float64),
        x_weights=jnp.asarray(op.x_weights, dtype=jnp.float64),
        n_xi_for_x=jnp.asarray(op.n_xi_for_x, dtype=jnp.int32),
    )
    offset = transport_matrix_from_state_vectors(
        layout, vgrid, FluxSurface.from_operator(op), SpeciesParams.from_operator(op),
        jnp.zeros((2, layout.total_size), dtype=jnp.float64),
        rhs_mode=3, delta=op.delta, alpha=op.alpha,
        g_hat=g_hat, i_hat=i_hat, iota=iota, b0_over_bbar=b0_over_bbar,
    )[0, 0]  # fmt: skip

    d11 = c * wdot(s5, f5) + offset
    cand_a = c * phi_minus + offset
    cand_b = c * phi_plus + offset
    lower = jnp.minimum(cand_a, cand_b)
    upper = jnp.maximum(cand_a, cand_b)
    gap = jnp.abs(upper - lower) / jnp.maximum(jnp.abs(d11), jnp.finfo(jnp.float64).tiny)
    return MonoenergeticD11Bounds(d11=d11, lower=lower, upper=upper, gap=gap)
