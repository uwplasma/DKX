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

import numpy as np

from .vmec_wout import VmecWout


def optional_jax_geometry_backend_status() -> dict[str, bool]:
    """Return whether optional JAX geometry backends are importable."""
    return {
        "vmec_jax": find_spec("vmec_jax") is not None,
        "booz_xform_jax": find_spec("booz_xform_jax") is not None,
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
    ``sfincs_jax`` convention ``(mode, radius)``.
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


__all__ = [
    "optional_jax_geometry_backend_status",
    "vmec_wout_from_wout_like",
]
