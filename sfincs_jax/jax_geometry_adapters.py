"""Optional adapters for JAX-native geometry producers.

The functions in this module intentionally avoid importing ``vmec_jax`` or
``booz_xform_jax`` at module import time. They operate on structural "wout-like"
objects so users can pass objects produced by optional differentiable geometry
pipelines without making those packages mandatory dependencies of ``sfincs_jax``.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .vmec_wout import VmecWout


def optional_jax_geometry_backend_status() -> dict[str, bool]:
    """Return whether optional JAX geometry backends are importable.

    The check is deliberately shallow: it reports importability without importing
    the packages, initializing devices, or changing JAX configuration.  This keeps
    CLI startup and normal CI independent of optional differentiable-geometry
    packages.
    """
    return {
        "vmec_jax": find_spec("vmec_jax") is not None,
        "booz_xform_jax": find_spec("booz_xform_jax") is not None,
    }


def optional_jax_geometry_backend_report() -> dict[str, Any]:
    """Return user-facing metadata for the optional JAX geometry lane.

    This is intentionally static apart from shallow backend importability checks.
    The report clarifies the differentiability boundary without importing optional
    packages or claiming end-to-end VMEC-boundary-to-transport gradients.
    """
    return {
        "backends": optional_jax_geometry_backend_status(),
        "runnable_paths": {
            "no_optional_dependencies": "--check-backends",
            "file_backed_setup": "--wout /path/to/wout.nc",
            "optional_in_memory_setup": "--vmec-case circular_tokamak",
        },
        "gradient_availability": {
            "spectral_scale_to_boozer_proxy": "available_when_optional_backends_installed",
            "vmec_file_io": "setup_only_not_differentiated",
            "vmec_fixed_boundary_solve": "setup_only_not_differentiated",
            "sfincs_vmec_file_adapter": "setup_only_not_differentiated",
            "sfincs_kinetic_transport_solve": "not_covered_by_this_lane",
        },
        "differentiated_graph": [
            "scaled VMEC-like spectral arrays",
            "booz_xform_jax",
            "sfincs_jax Boozer-spectrum proxy objective",
        ],
        "outside_differentiated_graph": [
            "VMEC file I/O",
            "vmec_jax example fixed-boundary setup",
            "sfincs_jax VMEC file adapters",
            "SFINCS kinetic transport solve",
        ],
        "claim": "geometry_proxy_gradient_gate_not_full_transport_gradient",
    }


def geometry_proxy_workflow_summary(
    *,
    provenance: str | None = None,
    requested_surface: float | None = None,
    selected_surface: float | None = None,
    boozer_resolution: dict[str, int] | None = None,
    grid_shape: dict[str, int] | None = None,
    scale: float | None = None,
    proxy_objective: float | None = None,
    autodiff_gradient: float | None = None,
    finite_difference_gradient: float | None = None,
    finite_difference_step: float | None = None,
    gradient_rtol: float = 5.0e-3,
    gradient_atol: float = 1.0e-7,
    backend_status: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Build a reusable provenance summary for the geometry-proxy workflow.

    The summary is intentionally conservative: it records the stages that are
    differentiable in the public vmec_jax/booz_xform_jax handoff, the optional
    packages needed to run that handoff, and a numerical gradient gate for the
    scalar geometry proxy when both autodiff and finite-difference gradients are
    supplied.  It never claims gradients through the SFINCS kinetic transport
    solve.
    """
    status = optional_jax_geometry_backend_status() if backend_status is None else dict(backend_status)
    gate: dict[str, Any] = {
        "status": "not_run",
        "autodiff_gradient": None,
        "finite_difference_gradient": None,
        "finite_difference_step": finite_difference_step,
        "absolute_error": None,
        "rtol": float(gradient_rtol),
        "atol": float(gradient_atol),
        "claim": "geometry_proxy_gradient_gate_only",
    }
    if autodiff_gradient is not None and finite_difference_gradient is not None:
        autodiff_value = float(autodiff_gradient)
        fd_value = float(finite_difference_gradient)
        abs_error = abs(autodiff_value - fd_value)
        tolerance = float(gradient_atol) + float(gradient_rtol) * abs(fd_value)
        gate.update(
            {
                "status": "pass" if abs_error <= tolerance else "fail",
                "autodiff_gradient": autodiff_value,
                "finite_difference_gradient": fd_value,
                "absolute_error": abs_error,
                "tolerance": tolerance,
            }
        )

    return {
        "workflow": "vmec_jax_to_boozer_sfincs_geometry_proxy",
        "provenance": {
            "source": provenance,
            "requested_surface": requested_surface,
            "selected_surface": selected_surface,
            "boozer_resolution": boozer_resolution,
            "grid_shape": grid_shape,
            "scale": scale,
        },
        "required_optional_dependencies": {
            "vmec_jax": {
                "importable": bool(status.get("vmec_jax", False)),
                "used_for": "VMEC-like wout provenance and optional fixed-boundary setup",
            },
            "booz_xform_jax": {
                "importable": bool(status.get("booz_xform_jax", False)),
                "used_for": "Boozer transform in the differentiable geometry-proxy path",
            },
        },
        "stages": [
            {
                "name": "vmec_provenance",
                "component": "vmec_jax or VMEC wout input",
                "differentiability": "setup_only_not_differentiated",
            },
            {
                "name": "spectral_scale",
                "component": "in-memory VMEC-like magnetic spectrum",
                "differentiability": "differentiated",
            },
            {
                "name": "boozer_transform",
                "component": "booz_xform_jax",
                "differentiability": "differentiated_for_geometry_proxy_gate",
            },
            {
                "name": "sfincs_geometry_proxy",
                "component": "sfincs_jax Boozer-spectrum proxy objective",
                "differentiability": "differentiated",
            },
            {
                "name": "sfincs_kinetic_transport_solve",
                "component": "full kinetic transport solver",
                "differentiability": "not_claimed_not_covered_by_this_lane",
            },
        ],
        "numerical_gradient_gate": gate,
        "results": {
            "proxy_objective": proxy_objective,
        },
        "claims": {
            "differentiable": (
                "scaled spectral arrays -> booz_xform_jax -> sfincs_jax "
                "Boozer-spectrum proxy objective"
            ),
            "not_claimed": "full VMEC-boundary-to-SFINCS kinetic transport gradients",
        },
    }


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
    """Convert a VMEC-like in-memory object into ``sfincs_jax.vmec_wout.VmecWout``.

    This covers the field names used by ``vmec_jax.wout.WoutData`` and by
    ``sfincs_jax.vmec_wout.VmecWout``. Arrays may be stored either as
    ``(radius, mode)`` or ``(mode, radius)``; the returned object always uses the
    ``sfincs_jax`` convention ``(mode, radius)``.  The optional ``path`` argument
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
    vmec_jax/booz_xform_jax workflow gates without making either package a
    required dependency of ``sfincs_jax``.
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


def boozer_spectrum_geometry_proxy_objective(
    bmnc_b: Any,
    ixm_b: Any,
    ixn_b: Any,
    *,
    theta: Any,
    zeta: Any,
    normalize: bool = True,
) -> jnp.ndarray:
    r"""Return a differentiable scalar proxy from a Boozer :math:`|B|` spectrum.

    This is a geometry/transport proxy, not a kinetic solve.  It measures the
    normalized field-strength variation plus a small angular-roughness penalty,
    giving optional vmec_jax/booz_xform_jax examples a bounded scalar that can be
    differentiated, finite-difference checked, and optimized quickly.
    """
    bhat = boozer_bhat_from_spectrum(
        theta,
        zeta,
        bmnc_b=bmnc_b,
        ixm_b=ixm_b,
        ixn_b=ixn_b,
        normalize=normalize,
    )
    centered = bhat - jnp.mean(bhat)
    theta_slope = jnp.mean((jnp.roll(bhat, -1, axis=0) - bhat) ** 2)
    zeta_slope = jnp.mean((jnp.roll(bhat, -1, axis=1) - bhat) ** 2)
    return jnp.mean(centered**2) + 0.05 * (theta_slope + zeta_slope)


__all__ = [
    "boozer_bhat_from_spectrum",
    "boozer_spectrum_geometry_proxy_objective",
    "geometry_proxy_workflow_summary",
    "optional_jax_geometry_backend_report",
    "optional_jax_geometry_backend_status",
    "vmec_wout_from_wout_like",
]
