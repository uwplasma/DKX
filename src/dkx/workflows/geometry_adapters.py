"""Optional adapters for JAX-native geometry producers.

The functions in this module intentionally avoid importing ``vmex`` or
``booz_xform_jax`` at module import time. They operate on structural "wout-like"
objects so users can pass objects produced by optional differentiable geometry
pipelines without making those packages mandatory dependencies of ``dkx``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from ..runtime import configure as _configure_runtime

_configure_runtime()

import jax.numpy as jnp
import numpy as np

from dkx.magnetic_geometry import VmecWout


def _attr(obj: Any, *names: str, default: Any = None, required: bool = True) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if required:
        joined = ", ".join(names)
        raise AttributeError(f"wout-like object is missing required attribute: {joined}")
    return default


def _as_1d(obj: Any, *names: str, dtype: Any) -> np.ndarray:
    arr = np.asarray(_attr(obj, *names), dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{names[0]} must be 1D, got shape {arr.shape}")
    return arr


def _mode_radius_array(
    value: Any,
    *,
    modes: int,
    radius_options: tuple[int, ...],
    name: str,
) -> np.ndarray:
    """Normalize a VMEC coefficient table to internal ``(mode, radius)`` layout."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.shape[0] == int(modes) and arr.shape[1] in radius_options:
        return arr
    if arr.shape[1] == int(modes) and arr.shape[0] in radius_options:
        return arr.T
    raise ValueError(
        f"{name} must have shape (modes, radius) or (radius, modes); "
        f"got {arr.shape}, modes={int(modes)}, radius_options={radius_options}"
    )


def _mode_radius_attr(
    obj: Any,
    name: str,
    *,
    modes: int,
    radius_options: tuple[int, ...],
    default_like: np.ndarray | None = None,
) -> np.ndarray:
    """Read and normalize a coefficient table, optionally zero-filling one table.

    Some in-memory VMEC producers expose a minimal stellarator-symmetric subset.
    For those objects, covariant/contravariant magnetic-field coefficient tables
    that are absent from the producer can be represented as zeros.  Required field
    strength, metric/Jacobian, and shape tables remain strict and raise if absent.
    """
    if hasattr(obj, name):
        value = getattr(obj, name)
    elif default_like is not None:
        value = np.zeros_like(default_like)
    else:
        raise AttributeError(f"wout-like object is missing required attribute: {name}")
    return _mode_radius_array(value, modes=modes, radius_options=radius_options, name=name)


def vmec_wout_from_wout_like(wout_like: Any, *, path: str | Path | None = None) -> VmecWout:
    """Convert a VMEC-like in-memory object into :class:`dkx.magnetic_geometry.VmecWout`.

    This covers the field names used by ``vmex.wout.WoutData`` and by
    :class:`dkx.magnetic_geometry.VmecWout`. Arrays may be stored either as
    ``(radius, mode)`` or ``(mode, radius)``; the returned object always uses the
    ``dkx`` convention ``(mode, radius)``.  The optional ``path`` argument
    is metadata only and is useful when the source object is produced entirely in
    memory.
    """
    ns = int(_attr(wout_like, "ns"))
    mnmax = int(_attr(wout_like, "mnmax", default=np.asarray(_attr(wout_like, "xm")).size, required=False))
    mnmax_nyq = int(
        _attr(wout_like, "mnmax_nyq", default=np.asarray(_attr(wout_like, "xm_nyq")).size, required=False)
    )
    path_value = Path(path) if path is not None else Path(str(_attr(wout_like, "path", default="<in-memory-wout>", required=False)))

    xm = _as_1d(wout_like, "xm", dtype=np.int32)
    xn = _as_1d(wout_like, "xn", dtype=np.int32)
    xm_nyq = _as_1d(wout_like, "xm_nyq", dtype=np.int32)
    xn_nyq = _as_1d(wout_like, "xn_nyq", dtype=np.int32)

    bmnc = _mode_radius_attr(wout_like, "bmnc", modes=mnmax_nyq, radius_options=(ns,))
    gmnc = _mode_radius_attr(wout_like, "gmnc", modes=mnmax_nyq, radius_options=(ns,))
    bsubumnc = _mode_radius_attr(wout_like, "bsubumnc", modes=mnmax_nyq, radius_options=(ns,), default_like=gmnc)
    bsubvmnc = _mode_radius_attr(wout_like, "bsubvmnc", modes=mnmax_nyq, radius_options=(ns,), default_like=gmnc)
    bsubsmns = _mode_radius_attr(wout_like, "bsubsmns", modes=mnmax_nyq, radius_options=(ns,), default_like=gmnc)
    bsupumnc = _mode_radius_attr(wout_like, "bsupumnc", modes=mnmax_nyq, radius_options=(ns,), default_like=gmnc)
    bsupvmnc = _mode_radius_attr(wout_like, "bsupvmnc", modes=mnmax_nyq, radius_options=(ns,), default_like=gmnc)

    rmnc = _mode_radius_attr(wout_like, "rmnc", modes=mnmax, radius_options=(ns,))
    zmns = _mode_radius_attr(wout_like, "zmns", modes=mnmax, radius_options=(ns,))
    lmns = _mode_radius_attr(wout_like, "lmns", modes=mnmax, radius_options=(ns, ns - 1))

    return VmecWout(
        path=path_value,
        nfp=int(_attr(wout_like, "nfp")),
        ns=ns,
        mpol=int(_attr(wout_like, "mpol")),
        ntor=int(_attr(wout_like, "ntor")),
        mnmax=mnmax,
        mnmax_nyq=mnmax_nyq,
        lasym=bool(_attr(wout_like, "lasym", default=False, required=False)),
        aminor_p=float(_attr(wout_like, "aminor_p", "Aminor_p", default=0.0, required=False)),
        phi=np.asarray(_attr(wout_like, "phi"), dtype=np.float64),
        xm=xm,
        xn=xn,
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
        bmnc=bmnc,
        gmnc=gmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        bsubsmns=bsubsmns,
        bsupumnc=bsupumnc,
        bsupvmnc=bsupvmnc,
        rmnc=rmnc,
        zmns=zmns,
        lmns=lmns,
        iotas=np.asarray(_attr(wout_like, "iotas"), dtype=np.float64),
        presf=np.asarray(_attr(wout_like, "presf", default=np.zeros((ns,), dtype=np.float64), required=False), dtype=np.float64),
    )


def boozer_bhat_from_spectrum(
    theta: Any,
    zeta: Any,
    *,
    bmnc_b: Any,
    ixm_b: Any,
    ixn_b: Any,
    normalize: bool = True,
    eps: float = 1.0e-14,
) -> jnp.ndarray:
    r"""Evaluate normalized magnetic-field strength from a Boozer cosine spectrum.

    Parameters use the public ``booz_xform_jax`` output convention: ``bmnc_b`` is a
    one-dimensional Boozer :math:`|B|` cosine spectrum, while ``ixm_b`` and
    ``ixn_b`` are the corresponding integer mode numbers.  In ``booz_xform_jax``
    output, ``ixn_b`` already includes the field-period factor, so this function
    evaluates :math:`m\theta - n\zeta` directly.

    The helper is intentionally small and pure JAX.  It is used by optional
    vmex/booz_xform_jax workflow gates without making either package a
    required dependency of ``dkx``.
    """
    theta_arr = jnp.asarray(theta)
    zeta_arr = jnp.asarray(zeta)
    coeff = jnp.asarray(bmnc_b)
    m_mode = jnp.asarray(ixm_b)
    n_mode = jnp.asarray(ixn_b)

    if coeff.ndim != 1:
        raise ValueError(f"bmnc_b must be 1D, got shape {coeff.shape}")
    if m_mode.ndim != 1 or n_mode.ndim != 1:
        raise ValueError("ixm_b and ixn_b must be 1D")
    if coeff.shape[0] != m_mode.shape[0] or coeff.shape[0] != n_mode.shape[0]:
        raise ValueError(
            "bmnc_b, ixm_b, and ixn_b must have the same length; "
            f"got {coeff.shape[0]}, {m_mode.shape[0]}, {n_mode.shape[0]}"
        )

    angle = (
        m_mode[:, None, None] * theta_arr[None, :, None]
        - n_mode[:, None, None] * zeta_arr[None, None, :]
    )
    bmod = jnp.sum(coeff[:, None, None] * jnp.cos(angle), axis=0)
    if not normalize:
        return bmod

    b00 = jnp.sum(jnp.where((m_mode == 0) & (n_mode == 0), coeff, 0.0))
    fallback = jnp.mean(jnp.abs(bmod)) + eps
    denom = jnp.where(jnp.abs(b00) > eps, b00, fallback)
    return bmod / denom


#: The operator leaves a flux-surface geometry owns.  Replacing exactly these
#: keeps the collision operator, grids, species and drives fixed, so a
#: derivative taken through them is a pure shape derivative.
BOOZER_SURFACE_GEOMETRY_LEAVES = (
    "b_hat", "db_hat_dtheta", "db_hat_dzeta", "d_hat",
    "b_hat_sup_theta", "b_hat_sup_zeta", "b_hat_sub_theta", "b_hat_sub_zeta",
)  # fmt: skip


def boozer_route_psi_a_hat(phi_edge: float, signgs: int, g_plus_iota_i: float) -> float:
    r"""``psiAHat`` for an operator built on a ``booz_xform`` surface of a VMEC equilibrium.

    This is the one place the handedness of the Boozer route is converted.
    :meth:`FluxSurfaceGeometry.from_fourier` builds the Boozer Jacobian as
    :math:`D = B^2/(G + \iota I)`, whose sign is that of :math:`G + \iota I`.
    The VMEC-file route (``geometryScheme = 5``) instead carries VMEC's own
    Jacobian, whose sign is ``signgs * sign(phi_edge)``, with
    ``psiAHat = phi_edge / (2 pi)``.  A drift-kinetic operator depends on
    the flux label only through ``D / psiAHat``-type products, so the two
    routes describe the same plasma when
    ``psiAHat = |phi_edge| / (2 pi) * signgs * sign(G + iota I)``.

    ``booz_xform`` keeps VMEC's angles: a ``signgs = -1`` equilibrium (every
    equilibrium VMEX writes) comes back with ``G > 0`` and the transform's sign
    of ``iota``, so ``psiAHat`` is negative on that route.  Taking
    ``+|phi_edge| / (2 pi)`` instead flips the sign of every flux and of
    ``FSABjHat``; ``tests/test_boozer_route_sign.py`` pins this against the
    VMEC-file route and the Redl formula.
    """
    return (
        abs(float(phi_edge)) / (2.0 * np.pi)
        * float(np.sign(signgs)) * float(np.sign(g_plus_iota_i))
    )  # fmt: skip


#: The operator leaves that carry ``1/psiAHat``: the density, temperature and
#: potential gradients in ``d/dpsiHat``.  Nothing else in a
#: :class:`~dkx.drift_kinetic.KineticOperator` depends on ``psiAHat``
#: (``tests/test_vmex_kinetic_gradient.py`` pins this), so the handedness of
#: the Boozer route can be converted on these leaves alone, inside a trace.
PSI_A_HAT_LEAVES = ("dn_hat_dpsi_hat", "dt_hat_dpsi_hat",
                    "dphi_hat_dpsi_hat", "dphi_hat_dpsi_hat_kinetic")  # fmt: skip


def convert_boozer_route_handedness(
    operator: Any, *, signgs: int, g_hat: Any, iota: Any, i_hat: Any
) -> Any:
    r"""Traced form of :func:`boozer_route_psi_a_hat` for an operator built at ``psiAHat > 0``.

    Multiplies the :data:`PSI_A_HAT_LEAVES` by ``signgs * sign(G + iota I)``,
    which is the same as building the operator with the signed
    ``psiAHat`` of :func:`boozer_route_psi_a_hat`.  ``G``, ``I`` and
    ``iota`` may be traced (the sign carries no derivative), so a template
    built once serves every trial equilibrium of an optimization.
    """
    from dataclasses import replace  # noqa: PLC0415

    sign = float(np.sign(signgs)) * jnp.sign(g_hat + iota * i_hat)
    return replace(operator, **{leaf: getattr(operator, leaf) * sign for leaf in PSI_A_HAT_LEAVES})


def kinetic_operator_on_boozer_surface(
    template: Any,
    *,
    bmnc_b: Any,
    ixm_b: Any,
    ixn_b: Any,
    nfp: int,
    iota: Any,
    g_hat: Any,
    i_hat: Any,
    signgs: int | None = None,
) -> Any:
    r"""Return ``template`` with its geometry replaced by one Boozer surface.

    ``bmnc_b``, ``ixm_b`` and ``ixn_b`` follow the ``booz_xform_jax`` output
    convention (``ixn_b`` carries the field-period factor); ``iota``, ``g_hat``
    and ``i_hat`` are that surface's ``iota_b``, ``bvco_b`` and ``buco_b`` in
    ``BBar = 1 T``, ``RBar = 1 m`` units.  Pure JAX in every amplitude and flux
    function, so a gradient flows from the kinetic solve back to the spectrum.
    The template's ``psiAHat`` must come from :func:`boozer_route_psi_a_hat`;
    or, with ``signgs`` given, be ``+|phi_edge|/(2 pi)``, and the handedness
    is converted here by :func:`convert_boozer_route_handedness`.
    """
    from dataclasses import replace  # noqa: PLC0415

    from dkx.magnetic_geometry import FluxSurfaceGeometry  # noqa: PLC0415

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, template.n_theta, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / int(nfp), template.n_zeta, endpoint=False)
    ixn = np.asarray(ixn_b)
    if np.any(ixn % int(nfp)):
        raise ValueError("ixn_b must carry the field-period factor (booz_xform convention)")
    surface = FluxSurfaceGeometry.from_fourier(
        theta=theta, zeta=zeta, bmnc=jnp.asarray(bmnc_b), m=jnp.asarray(np.asarray(ixm_b)),
        n=jnp.asarray(ixn // int(nfp)), n_periods=int(nfp), iota=iota, g_hat=g_hat, i_hat=i_hat,
    )  # fmt: skip
    leaves = {name: getattr(surface, name) for name in BOOZER_SURFACE_GEOMETRY_LEAVES}
    leaves["fsab_hat2"] = surface.fsab_hat2(
        theta_weights=template.theta_weights, zeta_weights=template.zeta_weights
    )
    operator = replace(template, **leaves)
    if signgs is None:
        return operator
    return convert_boozer_route_handedness(
        operator, signgs=signgs, g_hat=g_hat, iota=iota, i_hat=i_hat)


__all__ = [
    "BOOZER_SURFACE_GEOMETRY_LEAVES",
    "PSI_A_HAT_LEAVES",
    "convert_boozer_route_handedness",
    "boozer_bhat_from_spectrum",
    "boozer_route_psi_a_hat",
    "kinetic_operator_on_boozer_surface",
    "vmec_wout_from_wout_like",
]
