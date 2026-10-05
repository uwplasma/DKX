"""Momentum correction: a momentum-conserving bootstrap current from pitch-angle solves.

Pitch-angle scattering (PAS, ``collisionOperator = 1``) conserves particles but
not parallel momentum, so parallel flows and the bootstrap current computed with
it are biased.  The moment method of Sugama & Nishimura (Phys. Plasmas 9, 4637,
2002), used for stellarators by Maassberg, Beidler & Turkin (Phys. Plasmas 16,
072504, 2009), restores it: the parallel-flow (``l = 1``) part of the linearized
collision operator is replaced by its projection onto the Sonine flow moments
of every species -- the parallel particle flow and heat flow for two terms, the
13-moment approximation; three, the default, is Maassberg's ``j_x = 2`` -- the
rest stays PAS, and the coupled parallel momentum balance becomes an
``(N S) x (N S)`` linear system per surface for ``N`` Sonine terms and ``S``
species.

Formulation, in DKX's native units.  With ``K_PAS`` and ``K_FP`` the kinetic
operators of one deck under PAS and under the full linearized Fokker-Planck
operator, ``D = K_PAS - K_FP`` is the collisional part PAS drops (energy
scattering, the field-particle term, unlike-species friction); it is local in
space and diagonal in Legendre mode.  The model equation is

    K_PAS f = S + D P f,

the full operator on the Sonine part ``P f`` of the ``l = 1`` distribution and
PAS on the rest, as in Sugama & Nishimura's equations (13)-(16).  ``P f`` is
``sum_k alpha_ak(theta, zeta) phi_k`` with ``phi_k = x L_k^(3/2)(x^2) exp(-x^2)``,
the coefficients matched on the moments ``sum_x w nu_D x^3 L_k f``: weighting by
the PAS deflection frequency leaves the remainder ``f - P f`` without PAS
momentum, so the model conserves total parallel momentum exactly and the fluxes
of a symmetric field are intrinsically ambipolar.  ``D phi_k`` holds the
Hirshman-Sigmar friction coefficients ``l^ab_ij`` and the field-particle
momentum restoration, evaluated from DKX's Rosenbluth-potential operator
(:func:`friction_drives`).

The local Sonine flow is the surface part ``A_ak B / <B^2>`` plus a
Pfirsch-Schlueter part fixed by the gradients alone (Sugama & Nishimura
equation (11)), which the corrected state shares with the uncorrected PAS
solution ``g_0``.  So ``f = g_0 + p + sum_bk A_bk h_bk`` with ``h_bk`` the PAS
responses to the unit drives ``D phi_bk B/<B^2>`` (their surface flows are the
energy-convolved viscosity coefficients, the per-speed monoenergetic
convolution done on DKX's speed grid), ``p`` the response to the friction of
``g_0``'s Pfirsch-Schlueter flow, and

    (I - W) A = <B alpha(g_0 + p)>,      W_ij = <B alpha_i(h_j)>,

an ``(N S) x (N S)`` solve.  Every moment -- ``<j.B>``, particle and heat
fluxes, flows -- is read off the corrected state with the ordinary moment table,
so the radial fluxes carry their back-substituted correction.  The ``1 + N S``
right-hand sides of the first PAS solve share one elimination; ``p`` is one
more.  The route is traceable and differentiable.

Scope.  The radial electric field enters only through the PAS operator, valid
for ``E* << 1`` (Maassberg et al. 2009); and Mollen et al. (Phys. Plasmas 22,
112508, 2015) find a low-collisionality multi-species coefficient at ``E_r = 0``
that no moment correction reproduces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

__all__ = ["MomentumCorrectionResult", "friction_drives", "momentum_corrected_solve"]

_REMOVED = ("ParallelViscosity", "momentum_corrected_bootstrap", "parallel_friction_matrix",
            "parallel_viscosity", "solve_corrected_flows")  # fmt: skip


def __getattr__(name: str) -> Any:
    if name in _REMOVED:
        raise AttributeError(
            f"dkx.momentum_correction.{name} was removed: use momentum_corrected_solve("
            "pas_operator, fp_operator). The single-moment database correction lacked the "
            "heat-flow moment and so the temperature-gradient bootstrap drive (CHANGELOG)."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class MomentumCorrectionResult:
    """Momentum-corrected state and its pieces.

    Attributes:
        state: corrected solution, ``(total_size,)``; read moments from it with
            :func:`dkx.run.profile_moments_from_operator` and the PAS operator.
        uncorrected_state: the plain PAS solution ``g_0``.
        coefficients: ``(S, N)`` corrected Sonine flows ``<B alpha_ak>``.
        flows, uncorrected_flows: ``(S, N)`` ``<B alpha_ak>`` of ``state`` and ``g_0``.
        fsab_j, uncorrected_fsab_j: ``FSABjHat`` of ``state`` and of ``g_0``.
    """

    state: Any
    uncorrected_state: Any
    coefficients: Any
    flows: Any
    uncorrected_flows: Any
    fsab_j: Any
    uncorrected_fsab_j: Any


def _laguerre(k2: jnp.ndarray, n_sonine: int) -> jnp.ndarray:
    """Sonine polynomials ``L_k^(3/2)(x^2)``, ``k < n_sonine``, ``(N, X)``."""
    if n_sonine < 1:
        raise ValueError("n_sonine must be at least 1")
    terms = [jnp.ones_like(k2), 2.5 - k2]
    for k in range(1, n_sonine - 1):
        terms.append(((2 * k + 2.5 - k2) * terms[k] - (k + 1.5) * terms[k - 1]) / (k + 1))
    return jnp.stack(terms[:n_sonine])


def _sonine(op: Any, n_sonine: int) -> jnp.ndarray:
    """Basis ``phi_k = x L_k^(3/2)(x^2) exp(-x^2)`` on the speed grid, ``(N, X)``."""
    k2 = op.x * op.x
    return _laguerre(k2, n_sonine) * (op.x * jnp.exp(-k2))[None, :]


def _projection(op: Any, n_sonine: int) -> jnp.ndarray:
    """Test functions ``nu_D(x) x^3 L_k(x^2) w`` per species, ``(S, N, X)``.

    Weighting by the PAS deflection frequency makes the residual ``f - P f``
    carry no PAS momentum, so the model operator conserves total parallel
    momentum exactly and fluxes are intrinsically ambipolar in a symmetric field.
    ``nu_D`` is read off the PAS operator's ``l = 1`` diagonal on a uniform state.
    """
    ones = jnp.zeros(op.f_shape).at[:, :, 1].set(1.0)
    nu_d = op.apply_f(ones)[:, :, 1, 0, 0]  # (S, X); -C_PAS = nu_D on l = 1
    k2 = op.x * op.x
    return _laguerre(k2, n_sonine)[None] * (nu_d * op.x_weights * k2 * op.x)[:, None, :]


def _sonine_coefficients(op: Any, state: jnp.ndarray, test: jnp.ndarray) -> jnp.ndarray:
    """Local coefficients ``alpha`` of ``f_(l=1) ~ sum_k alpha_k phi_k``, ``(..., S, N, T, Z)``.

    Petrov-Galerkin: matched on the moments ``sum_x test_k f`` of :func:`_projection`.
    """
    f = state[..., : op.f_size].reshape(state.shape[:-1] + op.f_shape)[..., 1, :, :]
    gram_inv = jnp.linalg.inv(test @ _sonine(op, test.shape[1]).T)  # (S, k, j) -> inverse
    return jnp.einsum("sjk,...sxtz,skx->...sjtz", gram_inv, f, test)


def _fsa(op: Any, field: jnp.ndarray) -> jnp.ndarray:
    """Flux-surface average over the trailing ``(T, Z)`` axes."""
    w_tz = op.theta_weights[:, None] * op.zeta_weights[None, :] / op.d_hat
    return jnp.einsum("...tz,tz->...", field, w_tz) / jnp.sum(w_tz)


def _drive(op: Any, d_phi: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """State ``sum_bk d_phi_bk(a, x) alpha_bk(theta, zeta)`` on ``l = 1``, ``alpha (..., S, N, T, Z)``."""
    lead = alpha.shape[:-4]
    d = d_phi.reshape(alpha.shape[-4:-2] + d_phi.shape[1:])  # (b, k, a, x)
    f = jnp.zeros(lead + op.f_shape)
    f = f.at[..., 1, :, :].set(jnp.einsum("bkax,...bktz->...axtz", d, alpha))
    pad = jnp.zeros(lead + (op.total_size - op.f_size,))
    return jnp.concatenate([f.reshape(lead + (-1,)), pad], axis=-1)


def friction_drives(pas_operator: Any, fp_operator: Any, n_sonine: int = 3) -> jnp.ndarray:
    """``D phi_bk`` with ``D = K_PAS - K_FP``: the collisional drive of a unit Sonine flow.

    ``(S N, S, X)``: the ``l = 1`` speed profile that the part of the linearized
    Fokker-Planck operator PAS drops (friction ``l^ab_ij``, the field-particle
    momentum restoration, energy scattering) exerts on species ``a`` when
    species ``b`` carries the ``k``-th Sonine flow.  Collisions are local, so it
    depends on species and speed grid only: compute it once, on any geometry,
    and pass it as ``friction`` to :func:`momentum_corrected_solve`.
    """
    s_n = pas_operator.f_shape[0]
    phi = jnp.eye(s_n)[:, None, :, None] * _sonine(pas_operator, n_sonine)[None, :, None, :]
    phi = phi.reshape(s_n * n_sonine, s_n, -1)
    f = jnp.zeros(phi.shape[:1] + pas_operator.f_shape)
    f = f.at[:, :, :, 1].set(phi[:, :, :, None, None])
    d = jnp.stack([pas_operator.apply_f(g) - fp_operator.apply_f(g) for g in f])
    return d[:, :, :, 1, 0, 0]


def _refined_solve(op: Any, cols: jnp.ndarray, kw: dict, refine: int) -> jnp.ndarray:
    """``K_PAS^-1 cols`` with ``refine`` steps of iterative refinement, ``(n, m)``."""
    from .solve import solve  # noqa: PLC0415

    result = solve(op, cols, **kw)
    x = result.x.reshape(cols.shape)
    reuse = {} if result.factors is None else {"factors": result.factors}
    for _ in range(refine):
        residual = cols - jax.vmap(op.apply, in_axes=1, out_axes=1)(x)
        x = x + solve(op, residual, **kw, **reuse).x.reshape(cols.shape)
    return x


def momentum_corrected_solve(
    pas_operator: Any,
    fp_operator: Any = None,
    *,
    rhs: Any = None,
    friction: Any = None,
    n_sonine: int = 3,
    tol: float = 1.0e-10,
    differentiable: bool = True,
    refine: int | None = None,
) -> MomentumCorrectionResult:
    """Momentum-corrected PAS solve of one surface.

    Args:
        pas_operator: the deck's operator with ``collisionOperator = 1``.
        fp_operator: the same species and speed grid with ``collisionOperator = 0``,
            used only for the friction; not needed when ``friction`` is given.
        rhs: drive, default ``pas_operator.rhs()``.
        friction: cached :func:`friction_drives` (same ``n_sonine``).
        n_sonine: Sonine terms; 2 is the 13-moment method of Sugama & Nishimura,
            3 the ``j_x = 2`` moment technique of Maassberg et al.
        tol, differentiable: passed to :func:`dkx.solve.solve`.
        refine: iterative-refinement steps on the PAS solves, default 1 for a
            non-differentiable call and 0 otherwise (the electron Sonine drives
            are steep in speed; the structured elimination meets them to ~1e-4
            at low collisionality, and refinement moves ``<j.B>`` by ~1e-7).
    """
    from .run import profile_moments_from_operator  # noqa: PLC0415

    op = pas_operator
    d_phi = friction if friction is not None else friction_drives(op, fp_operator, n_sonine)
    s_n = op.f_shape[0]
    # The structured elimination meets the steep electron drives to ~1e-4 at
    # low collisionality; refinement, not a many-column Krylov fallback, closes it.
    refine = (0 if differentiable else 1) if refine is None else int(refine)
    loose = tol if differentiable or refine == 0 else max(tol, 1e-3)
    kw = dict(method="auto", tol=loose, differentiable=differentiable,
              tier1_keep_lowest=op.n_xi, emit=None)  # fmt: skip
    b2 = _fsa(op, op.b_hat**2)
    # One column per unit B-shaped Sonine flow, alpha_bk = B / <B^2>.
    unit = jnp.eye(s_n * n_sonine).reshape(-1, s_n, n_sonine)[..., None, None] * op.b_hat / b2
    rhs0 = op.rhs() if rhs is None else jnp.asarray(rhs)
    cols = jnp.concatenate([rhs0[:, None], _drive(op, d_phi, unit).T], axis=1)
    sol = _refined_solve(op, cols, kw, refine)
    g0, h = sol[:, 0], sol[:, 1:].T

    test = _projection(op, n_sonine)

    def flow(state: jnp.ndarray) -> jnp.ndarray:  # <B alpha>, (..., S, N)
        return _fsa(op, op.b_hat * _sonine_coefficients(op, state, test))

    # The Pfirsch-Schlueter part of the flow is set by the gradients alone, so
    # the corrected state shares g0's; its friction is a known drive.
    alpha0 = _sonine_coefficients(op, g0, test)
    alpha_ps = alpha0 - op.b_hat * flow(g0)[..., None, None] / b2
    p = _refined_solve(op, _drive(op, d_phi, alpha_ps)[:, None], kw, refine)[:, 0]
    w = flow(h).reshape(len(h), -1).T  # W_ij = <B alpha_i(h_j)>
    a = jnp.linalg.solve(jnp.eye(len(h)) - w, (flow(g0) + flow(p)).reshape(-1))
    f = g0 + p + a @ h
    return MomentumCorrectionResult(
        state=f, uncorrected_state=g0, coefficients=a.reshape(s_n, n_sonine),
        flows=flow(f), uncorrected_flows=flow(g0),
        fsab_j=profile_moments_from_operator(op, f)["FSABjHat"].reshape(()),
        uncorrected_fsab_j=profile_moments_from_operator(op, g0)["FSABjHat"].reshape(()),
    )  # fmt: skip
