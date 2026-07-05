"""Flat SFINCS output readers and writers.

This module owns the on-disk format boundary for small and moderate
``sfincsOutput`` payloads. Solver and diagnostic code should construct a plain
dataset dictionary; this layer serializes it to HDF5, NetCDF4, or NPZ while
preserving the SFINCS Fortran readback layout convention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
import hashlib
import os
import json
import math
import re

import h5py
import numpy as np

from ..namelist import Namelist
from sfincs_jax.discretization.v3 import V3Grids
from ..solvers.diagnostics import SolverTrace, write_solver_trace_h5


OUTPUT_GEOM_CACHE: dict[tuple[object, ...], dict[str, np.ndarray]] = {}
OUTPUT_GEOM_CACHE_VERSION = 2
OUTPUT_CACHE_FIELDS = (
    "gpsiHatpsiHat",
    "uHat",
    "diotadpsiHat",
    "VPrimeHat",
    "FSABHat2",
    "BDotCurlB",
    "classicalParticleFluxNoPhi1_psiHat",
    "classicalParticleFluxNoPhi1_psiN",
    "classicalParticleFluxNoPhi1_rHat",
    "classicalParticleFluxNoPhi1_rN",
    "classicalHeatFluxNoPhi1_psiHat",
    "classicalHeatFluxNoPhi1_psiN",
    "classicalHeatFluxNoPhi1_rHat",
    "classicalHeatFluxNoPhi1_rN",
)


def output_cache_enabled() -> bool:
    """Return whether geometry output caching is enabled."""

    cache_env = os.environ.get("SFINCS_JAX_OUTPUT_CACHE", "").strip().lower()
    if cache_env in {"0", "false", "no", "off"}:
        return False
    persist_env = os.environ.get("SFINCS_JAX_OUTPUT_CACHE_PERSIST", "").strip().lower()
    if persist_env in {"0", "false", "no", "off"}:
        return False
    return True


def output_cache_dir() -> Path | None:
    """Return and create the configured geometry-output cache directory."""

    cache_dir_env = os.environ.get("SFINCS_JAX_OUTPUT_CACHE_DIR", "").strip()
    if cache_dir_env:
        cache_dir = Path(cache_dir_env).expanduser()
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
        if xdg_cache:
            cache_dir = Path(xdg_cache) / "sfincs_jax" / "output_cache"
        else:
            cache_dir = Path.home() / ".cache" / "sfincs_jax" / "output_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return cache_dir


def output_cache_path(cache_key: tuple[object, ...]) -> Path | None:
    """Return the stable on-disk cache path for a geometry-output cache key."""

    cache_dir = output_cache_dir()
    if cache_dir is None:
        return None
    digest = hashlib.blake2b(repr(cache_key).encode("utf-8"), digest_size=16).hexdigest()
    return cache_dir / f"output_geom_{digest}.npz"


def hashable_value(val: Any) -> object:
    """Convert nested namelist values to stable, hashable cache-key pieces."""

    if isinstance(val, list):
        return tuple(hashable_value(v) for v in val)
    if isinstance(val, dict):
        return tuple(sorted((str(k), hashable_value(v)) for k, v in val.items()))
    return val


def group_subset_key(group: dict, keys: tuple[str, ...]) -> tuple[tuple[str, object], ...]:
    """Return a cache-key fragment for selected namelist group keys."""

    return tuple((key, hashable_value(group.get(key, None))) for key in keys)


@lru_cache(maxsize=256)
def file_content_identity(path_resolved: str, mtime_ns: int, file_size: int) -> tuple[int, str]:
    """Return a content hash keyed by file metadata for equilibrium cache keys."""

    h = hashlib.blake2b(digest_size=16)
    with Path(path_resolved).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return int(file_size), h.hexdigest()


def equilibrium_cache_identity(path: str | Path) -> tuple[int, str] | None:
    """Return a stable equilibrium-file identity, or ``None`` when unavailable."""

    try:
        p = Path(path).expanduser().resolve()
        st = p.stat()
    except OSError:
        return None
    return file_content_identity(str(p), int(st.st_mtime_ns), int(st.st_size))


def output_geom_cache_key(
    *,
    nml: Namelist,
    grids: V3Grids,
    get_int: Callable[[dict, str, int], int],
    resolve_equilibrium_file: Callable[..., Path] | None,
) -> tuple[object, ...] | None:
    """Return the geometry-output cache key for one namelist/grid pair.

    ``resolve_equilibrium_file`` is injected by ``io.py`` because equilibrium
    path localization is part of input orchestration rather than the output
    cache domain.
    """

    geom_params = nml.group("geometryParameters")
    species_params = nml.group("speciesParameters")
    phys_params = nml.group("physicsParameters")
    general_params = nml.group("general")
    geometry_scheme = int(get_int(geom_params, "geometryScheme", -1))
    eq_key = None
    if geometry_scheme in {5, 11, 12} and resolve_equilibrium_file is not None:
        try:
            eq_path = resolve_equilibrium_file(nml=nml)
            eq_key = equilibrium_cache_identity(eq_path)
        except Exception:  # noqa: BLE001
            eq_key = None
    equilibrium_keys = {
        "EQUILIBRIUMFILE",
        "FORT996BOOZER_FILE",
        "JGBOOZER_FILE",
        "JGBOOZER_FILE_NONSTELSYM",
    }
    geom_key_items = []
    for key, value in geom_params.items():
        key_u = str(key).upper()
        if eq_key is not None and key_u in equilibrium_keys:
            geom_key_items.append((key_u, ("__equilibrium__", eq_key)))
        else:
            geom_key_items.append((key_u, hashable_value(value)))
    geom_key = tuple(sorted(geom_key_items))
    grid_key = (int(grids.theta.size), int(grids.zeta.size))
    species_key = tuple(sorted((str(k).upper(), hashable_value(v)) for k, v in species_params.items()))
    phys_subset_key = group_subset_key(
        phys_params,
        (
            "DELTA",
            "ALPHA",
            "NU_N",
            "NUPRIME",
            "ESTAR",
        ),
    )
    general_subset_key = group_subset_key(general_params, ("RHSMODE",))
    return ("output_geom", geometry_scheme, geom_key, eq_key, grid_key, species_key, phys_subset_key, general_subset_key)


def load_output_cache(cache_key: tuple[object, ...]) -> dict[str, np.ndarray] | None:
    """Load a geometry-output cache payload from disk."""

    if not output_cache_enabled():
        return None
    path = output_cache_path(cache_key)
    if path is None or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(np.asarray(data.get("cache_version", 0)).reshape(())) != OUTPUT_GEOM_CACHE_VERSION:
                return None
            return {k: np.asarray(data[k]) for k in data.files if k in OUTPUT_CACHE_FIELDS}
    except Exception:  # noqa: BLE001
        return None


def save_output_cache(cache_key: tuple[object, ...], payload: dict[str, np.ndarray]) -> None:
    """Persist allowed geometry-output cache fields to disk."""

    if not output_cache_enabled():
        return
    path = output_cache_path(cache_key)
    if path is None:
        return
    try:
        data = {"cache_version": np.asarray(OUTPUT_GEOM_CACHE_VERSION, dtype=np.int32)}
        for field in OUTPUT_CACHE_FIELDS:
            if field in payload:
                data[field] = np.asarray(payload[field])
        np.savez_compressed(path, **data)
    except Exception:  # noqa: BLE001
        return


def decode_if_bytes(x: Any) -> Any:
    """Decode byte strings commonly returned by HDF5/NPZ/NetCDF readers."""

    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray) and x.dtype.kind in {"S", "U", "O"}:
        # Common case in SFINCS: 1-element byte-string array.
        if x.size == 1:
            item = x.reshape(-1)[0]
            return decode_if_bytes(item)
    return x


def read_sfincs_h5(path: Path) -> dict[str, Any]:
    """Read a SFINCS ``sfincsOutput.h5`` file into memory."""

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                v = obj[...]
                v = decode_if_bytes(v)
                out[name] = v

        f.visititems(visit)
    return out


def to_numpy_for_h5(x: Any) -> Any:
    """Return NumPy arrays for array-like values without importing JAX."""

    if isinstance(x, np.ndarray):
        return x
    # Handle JAX arrays without importing jax as a hard dependency at import time.
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return x


def fortran_h5_layout(x: Any) -> Any:
    """Mimic the layout of arrays written by SFINCS v3 Fortran HDF5 output."""

    arr = to_numpy_for_h5(x)
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.ndim <= 1:
        return arr
    axes = tuple(reversed(range(arr.ndim)))
    return np.ascontiguousarray(np.transpose(arr, axes=axes))


@dataclass(frozen=True)
class ExportFConfig:
    """Axis maps and metadata for Fortran-compatible ``export_f`` output."""

    export_full_f: bool
    export_delta_f: bool
    theta_option: int
    zeta_option: int
    x_option: int
    xi_option: int
    export_theta: np.ndarray
    export_zeta: np.ndarray
    export_x: np.ndarray
    export_xi: Optional[np.ndarray]
    n_export_theta: int
    n_export_zeta: int
    n_export_x: int
    n_export_xi: int
    map_theta: np.ndarray
    map_zeta: np.ndarray
    map_x: np.ndarray
    map_xi: np.ndarray


def get_namelist_float(group: dict, key: str, default: float) -> float:
    """Return a scalar namelist value as ``float`` with SFINCS key casing."""

    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v)


def get_namelist_int(group: dict, key: str, default: int) -> int:
    """Return a scalar namelist value as ``int`` with SFINCS key casing."""

    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return int(v)


def fortran_logical(value: bool) -> np.int32:
    """Return the SFINCS v3 convention for logical output datasets."""

    return np.int32(1 if bool(value) else -1)


_format_get_float = get_namelist_float
_format_get_int = get_namelist_int


def _as_1d_float(group: dict, key: str, *, default: float | None = None) -> np.ndarray:
    """Return a Fortran-namelist value as a 1-D ``float64`` array."""

    k = key.upper()
    if k not in group:
        if default is None:
            raise KeyError(key)
        return np.atleast_1d(np.asarray([default], dtype=np.float64))
    v = group[k]
    return np.atleast_1d(np.asarray(v, dtype=np.float64))


def _legendre_matrix(xi: np.ndarray, *, n_l: int) -> np.ndarray:
    """Evaluate Legendre polynomials ``P_0`` through ``P_{n_l-1}`` at ``xi``."""

    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    if n_l < 1:
        raise ValueError("n_l must be >= 1")
    out = np.zeros((xi.size, n_l), dtype=np.float64)
    out[:, 0] = 1.0
    if n_l == 1:
        return out
    out[:, 1] = xi
    for ell in range(2, n_l):
        out[:, ell] = ((2 * ell - 1) * xi * out[:, ell - 1] - (ell - 1) * out[:, ell - 2]) / float(ell)
    return out


def _export_f_config(*, nml: Any, grids: Any, geom: Any) -> ExportFConfig | None:
    """Build Fortran-compatible distribution-function export maps."""

    export_f = nml.group("export_f")
    export_full_f = bool(export_f.get("EXPORT_FULL_F", False))
    export_delta_f = bool(export_f.get("EXPORT_DELTA_F", False))
    if not (export_full_f or export_delta_f):
        return None

    # Fortran defaults from export_f.F90.
    theta_option = _format_get_int(export_f, "EXPORT_F_THETA_OPTION", 2)
    zeta_option = _format_get_int(export_f, "EXPORT_F_ZETA_OPTION", 2)
    xi_option = _format_get_int(export_f, "EXPORT_F_XI_OPTION", 1)
    x_option = _format_get_int(export_f, "EXPORT_F_X_OPTION", 0)

    export_theta = _as_1d_float(export_f, "EXPORT_F_THETA", default=0.0)
    export_zeta = _as_1d_float(export_f, "EXPORT_F_ZETA", default=0.0)
    export_xi = _as_1d_float(export_f, "EXPORT_F_XI", default=0.0)
    export_x = _as_1d_float(export_f, "EXPORT_F_X", default=1.0)

    theta = np.asarray(grids.theta, dtype=np.float64)
    zeta = np.asarray(grids.zeta, dtype=np.float64)
    x = np.asarray(grids.x, dtype=np.float64)

    n_theta = int(theta.size)
    n_zeta = int(zeta.size)
    n_x = int(x.size)
    n_xi = int(grids.n_xi)

    if theta_option == 0:
        export_theta = theta.copy()
        map_theta = np.eye(n_theta, dtype=np.float64)
    elif theta_option == 1:
        export_theta = np.mod(export_theta, 2.0 * math.pi)
        map_theta = np.zeros((export_theta.size, n_theta), dtype=np.float64)
        for j, val in enumerate(export_theta):
            idx1 = int(math.floor(val * n_theta / (2.0 * math.pi))) + 1
            if idx1 < 1:
                raise ValueError(f"Invalid export_f_theta index for value {val}")
            if idx1 == n_theta + 1:
                idx1 = n_theta
                idx2 = 1
            elif idx1 == n_theta:
                idx2 = 1
            elif idx1 > n_theta + 1:
                raise ValueError(f"Invalid export_f_theta index for value {val}")
            else:
                idx2 = idx1 + 1
            weight1 = idx1 - val * n_theta / (2.0 * math.pi)
            weight2 = 1.0 - weight1
            map_theta[j, idx1 - 1] = weight1
            map_theta[j, idx2 - 1] = weight2
    elif theta_option == 2:
        export_theta = np.mod(export_theta, 2.0 * math.pi)
        include = np.zeros((n_theta,), dtype=bool)
        for val in export_theta:
            err = np.minimum.reduce(
                [(val - theta) ** 2, (val - theta - 2.0 * math.pi) ** 2, (val - theta + 2.0 * math.pi) ** 2]
            )
            include[int(np.argmin(err))] = True
        export_theta = theta[include].copy()
        map_theta = np.zeros((export_theta.size, n_theta), dtype=np.float64)
        rows = np.where(include)[0]
        for row_idx, j in enumerate(rows):
            map_theta[row_idx, j] = 1.0
    else:
        raise ValueError("Invalid export_f_theta_option")

    if n_zeta == 1:
        export_zeta = np.asarray([0.0], dtype=np.float64)
        map_zeta = np.ones((1, 1), dtype=np.float64)
    else:
        zeta_period = 2.0 * math.pi / float(geom.n_periods)
        if zeta_option == 0:
            export_zeta = zeta.copy()
            map_zeta = np.eye(n_zeta, dtype=np.float64)
        elif zeta_option == 1:
            export_zeta = np.mod(export_zeta, zeta_period)
            map_zeta = np.zeros((export_zeta.size, n_zeta), dtype=np.float64)
            for j, val in enumerate(export_zeta):
                idx1 = int(math.floor(val * n_zeta / zeta_period)) + 1
                if idx1 < 1:
                    raise ValueError(f"Invalid export_f_zeta index for value {val}")
                if idx1 == n_zeta + 1:
                    idx1 = n_zeta
                    idx2 = 1
                elif idx1 == n_zeta:
                    idx2 = 1
                elif idx1 > n_zeta + 1:
                    raise ValueError(f"Invalid export_f_zeta index for value {val}")
                else:
                    idx2 = idx1 + 1
                weight1 = idx1 - val * n_zeta / zeta_period
                weight2 = 1.0 - weight1
                map_zeta[j, idx1 - 1] = weight1
                map_zeta[j, idx2 - 1] = weight2
        elif zeta_option == 2:
            export_zeta = np.mod(export_zeta, zeta_period)
            include = np.zeros((n_zeta,), dtype=bool)
            for val in export_zeta:
                err = np.minimum.reduce(
                    [(val - zeta) ** 2, (val - zeta - zeta_period) ** 2, (val - zeta + zeta_period) ** 2]
                )
                include[int(np.argmin(err))] = True
            export_zeta = zeta[include].copy()
            map_zeta = np.zeros((export_zeta.size, n_zeta), dtype=np.float64)
            rows = np.where(include)[0]
            for row_idx, j in enumerate(rows):
                map_zeta[row_idx, j] = 1.0
        else:
            raise ValueError("Invalid export_f_zeta_option")

    if x_option == 0:
        export_x = x.copy()
        map_x = np.eye(n_x, dtype=np.float64)
    elif x_option == 1:
        from ..physics.collisions import polynomial_interpolation_matrix_np  # noqa: PLC0415

        other = nml.group("otherNumericalParameters")
        x_grid_scheme = _format_get_int(other, "XGRIDSCHEME", _format_get_int(other, "xGridScheme", 5))
        x_grid_k = float(_format_get_float(other, "xGrid_k", 0.0))
        if x_grid_scheme not in {1, 2, 5, 6}:
            raise NotImplementedError(
                f"export_f_x_option=1 is only implemented for xGridScheme in {{1,2,5,6}} (got {x_grid_scheme})."
            )
        alpxk = np.exp(-(x * x)) * (x**x_grid_k)
        alpx = np.exp(-(export_x * export_x)) * (export_x**x_grid_k)
        map_x = polynomial_interpolation_matrix_np(xk=x, x=export_x, alpxk=alpxk, alpx=alpx)
    elif x_option == 2:
        include = np.zeros((n_x,), dtype=bool)
        for val in export_x:
            err = (val - x) ** 2
            include[int(np.argmin(err))] = True
        export_x = x[include].copy()
        map_x = np.zeros((export_x.size, n_x), dtype=np.float64)
        rows = np.where(include)[0]
        for row_idx, j in enumerate(rows):
            map_x[row_idx, j] = 1.0
    else:
        raise ValueError("Invalid export_f_x_option")

    if xi_option == 0:
        map_xi = np.eye(n_xi, dtype=np.float64)
        export_xi_out: Optional[np.ndarray] = None
        n_export_xi = n_xi
    elif xi_option == 1:
        map_xi = _legendre_matrix(export_xi, n_l=n_xi)
        export_xi_out = export_xi.copy()
        n_export_xi = int(export_xi.size)
    else:
        raise ValueError("Invalid export_f_xi_option")

    return ExportFConfig(
        export_full_f=export_full_f,
        export_delta_f=export_delta_f,
        theta_option=int(theta_option),
        zeta_option=int(zeta_option),
        x_option=int(x_option),
        xi_option=int(xi_option),
        export_theta=np.asarray(export_theta, dtype=np.float64),
        export_zeta=np.asarray(export_zeta, dtype=np.float64),
        export_x=np.asarray(export_x, dtype=np.float64),
        export_xi=export_xi_out,
        n_export_theta=int(export_theta.size),
        n_export_zeta=int(export_zeta.size),
        n_export_x=int(export_x.size),
        n_export_xi=int(n_export_xi),
        map_theta=map_theta,
        map_zeta=map_zeta,
        map_x=map_x,
        map_xi=map_xi,
    )


def _apply_export_f_maps(f: np.ndarray, cfg: ExportFConfig) -> np.ndarray:
    """Apply ``export_f`` maps to a distribution in ``(S, X, L, theta, zeta)`` order."""

    f = np.asarray(f, dtype=np.float64)
    f = np.einsum("ax,sxltz->saltz", cfg.map_x, f, optimize=True)
    f = np.einsum("bl,saltz->sabtz", cfg.map_xi, f, optimize=True)
    f = np.einsum("ct,sabtz->sabcz", cfg.map_theta, f, optimize=True)
    f = np.einsum("dz,sabcz->sabcd", cfg.map_zeta, f, optimize=True)
    return f


def write_export_f_state_vectors_to_data(
    *,
    data: dict[str, Any],
    state_vectors: list[Any] | tuple[Any, ...],
    f_size: int,
    f_shape: tuple[int, ...],
    f0_l0: Any,
    export_cfg: ExportFConfig | None,
    fortran_h5_layout_fn=fortran_h5_layout,
) -> None:
    """Write ``delta_f`` and ``full_f`` datasets for solved state vectors.

    The input distribution is in solver order ``(species, x, ell, theta, zeta)``.
    The stored pre-layout order matches the SFINCS Fortran readback convention:
    ``(x_export, xi_export, zeta_export, theta_export, species, iteration)``.
    """

    if export_cfg is None:
        return
    export_full_f = int(np.asarray(data.get("export_full_f", 0)).reshape(())) == 1
    export_delta_f = int(np.asarray(data.get("export_delta_f", 0)).reshape(())) == 1
    if not (export_full_f or export_delta_f) or not state_vectors:
        return

    f0_l0_arr = np.asarray(f0_l0, dtype=np.float64)
    delta_list: list[np.ndarray] = []
    full_list: list[np.ndarray] = []
    for x_full in state_vectors:
        f_delta = np.asarray(x_full[: int(f_size)], dtype=np.float64).reshape(f_shape)
        if export_delta_f:
            delta_np = _apply_export_f_maps(f_delta, export_cfg)
            delta_list.append(np.transpose(delta_np, (1, 2, 4, 3, 0)))
        if export_full_f:
            f_full = np.array(f_delta, dtype=np.float64, copy=True)
            f_full[:, :, 0, :, :] += f0_l0_arr
            full_np = _apply_export_f_maps(f_full, export_cfg)
            full_list.append(np.transpose(full_np, (1, 2, 4, 3, 0)))

    if export_delta_f and delta_list:
        data["delta_f"] = fortran_h5_layout_fn(np.stack(delta_list, axis=-1))
    if export_full_f and full_list:
        data["full_f"] = fortran_h5_layout_fn(np.stack(full_list, axis=-1))


def write_sfincs_h5(
    *,
    path: Path,
    data: dict[str, Any],
    fortran_layout: bool = True,
    overwrite: bool = True,
    solver_trace: SolverTrace | None = None,
) -> None:
    """Write a minimal SFINCS-style HDF5 file with flat datasets at root."""

    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for k, v in data.items():
            if v is None:
                continue
            vv = to_numpy_for_h5(v)
            if fortran_layout:
                vv = fortran_h5_layout(vv)
            if isinstance(vv, np.ndarray) and vv.ndim > 0 and vv.dtype.kind != "O":
                vv = np.ascontiguousarray(vv)
            f.create_dataset(k, data=vv)
        if solver_trace is not None:
            write_solver_trace_h5(f, solver_trace)


def output_file_format(path: Path) -> str:
    """Infer the on-disk output format from a filename suffix."""

    suffix = Path(path).suffix.lower()
    if suffix in {".h5", ".hdf5", ""}:
        return "h5"
    if suffix in {".nc", ".netcdf", ".cdf"}:
        return "netcdf"
    if suffix == ".npz":
        return "npz"
    raise ValueError(
        f"Unsupported sfincs_jax output suffix {suffix!r}. "
        "Use .h5/.hdf5, .nc/.netcdf, or .npz."
    )


def netcdf_safe_name(name: str, used: set[str]) -> str:
    """Return a NetCDF variable name while preserving the original name in attrs."""

    candidate = re.sub(r"[^0-9A-Za-z_]", "_", str(name)).strip("_")
    if not candidate:
        candidate = "dataset"
    if candidate[0].isdigit():
        candidate = f"v_{candidate}"
    base = candidate
    i = 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def write_sfincs_netcdf(
    *,
    path: Path,
    data: dict[str, Any],
    fortran_layout: bool = True,
    overwrite: bool = True,
    solver_trace: SolverTrace | None = None,
) -> None:
    """Write SFINCS datasets to an uncompressed NetCDF4 file."""

    try:
        from netCDF4 import Dataset  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - dependency is required by pyproject.
        raise RuntimeError("NetCDF output requires the netCDF4 package.") from exc

    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    original_names: dict[str, str] = {}
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("sfincs_jax_format", "netcdf")
        if solver_trace is not None:
            ds.setncattr("sfincs_jax_solver_trace_json", solver_trace.to_json())
        for key, value in data.items():
            if value is None:
                continue
            name = netcdf_safe_name(str(key), used)
            original_names[name] = str(key)
            vv = to_numpy_for_h5(value)
            if fortran_layout:
                vv = fortran_h5_layout(vv)
            arr = np.asarray(vv)
            dims: tuple[str, ...] = ()
            if arr.ndim:
                dims = tuple(f"{name}_dim_{i}" for i in range(arr.ndim))
                for dim_name, size in zip(dims, arr.shape, strict=False):
                    ds.createDimension(dim_name, int(size))
            if arr.dtype.kind in {"S", "U", "O"}:
                var = ds.createVariable(name, str, dims)
                if arr.ndim == 0:
                    var[...] = decode_if_bytes(arr.reshape(()).item())
                else:
                    var[...] = np.vectorize(decode_if_bytes, otypes=[str])(arr)
                continue
            if arr.dtype.kind == "b":
                arr = arr.astype(np.int8)
                original_dtype = "bool"
            else:
                original_dtype = str(arr.dtype)
            if arr.dtype.byteorder not in {"=", "|"}:
                arr = arr.astype(arr.dtype.type, copy=True)
            if arr.ndim > 0:
                arr = np.ascontiguousarray(arr)
            var = ds.createVariable(name, arr.dtype, dims, zlib=False)
            var.setncattr("sfincs_jax_original_dtype", original_dtype)
            if arr.ndim == 0:
                var[...] = arr.reshape(()).item()
            else:
                var[...] = arr
        ds.setncattr("sfincs_jax_original_names_json", json.dumps(original_names, sort_keys=True))


def write_sfincs_npz(
    *,
    path: Path,
    data: dict[str, Any],
    fortran_layout: bool = True,
    overwrite: bool = True,
    solver_trace: SolverTrace | None = None,
) -> None:
    """Write SFINCS datasets to a fast, uncompressed ``.npz`` archive."""

    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if value is None:
            continue
        vv = to_numpy_for_h5(value)
        if fortran_layout:
            vv = fortran_h5_layout(vv)
        arr = np.asarray(vv)
        if arr.ndim > 0 and arr.dtype.kind != "O":
            arr = np.ascontiguousarray(arr)
        payload[str(key)] = arr
    if solver_trace is not None:
        payload["sfincs_jax_solver_trace_json"] = np.asarray(solver_trace.to_json())
    np.savez(path, **payload)


def write_sfincs_output_file(
    *,
    path: Path,
    data: dict[str, Any],
    fortran_layout: bool = True,
    overwrite: bool = True,
    solver_trace: SolverTrace | None = None,
) -> None:
    """Write SFINCS output using the format selected by ``path``."""

    fmt = output_file_format(path)
    if fmt == "h5":
        write_sfincs_h5(
            path=path,
            data=data,
            fortran_layout=fortran_layout,
            overwrite=overwrite,
            solver_trace=solver_trace,
        )
    elif fmt == "netcdf":
        write_sfincs_netcdf(
            path=path,
            data=data,
            fortran_layout=fortran_layout,
            overwrite=overwrite,
            solver_trace=solver_trace,
        )
    elif fmt == "npz":
        write_sfincs_npz(
            path=path,
            data=data,
            fortran_layout=fortran_layout,
            overwrite=overwrite,
            solver_trace=solver_trace,
        )
    else:  # pragma: no cover - guarded by output_file_format.
        raise ValueError(fmt)


def read_sfincs_output_file(path: Path) -> dict[str, Any]:
    """Read ``.h5``, ``.nc``/``.netcdf``, or ``.npz`` SFINCS output into memory."""

    path = Path(path).resolve()
    fmt = output_file_format(path)
    if fmt == "h5":
        return read_sfincs_h5(path)
    if fmt == "npz":
        if not path.exists():
            raise FileNotFoundError(str(path))
        out: dict[str, Any] = {}
        with np.load(path, allow_pickle=False) as npz:
            for key in npz.files:
                out[key] = decode_if_bytes(npz[key])
        return out
    if fmt == "netcdf":
        try:
            from netCDF4 import Dataset  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - dependency is required by pyproject.
            raise RuntimeError("NetCDF output requires the netCDF4 package.") from exc
        if not path.exists():
            raise FileNotFoundError(str(path))
        out: dict[str, Any] = {}
        with Dataset(path, "r") as ds:
            mapping_raw = getattr(ds, "sfincs_jax_original_names_json", "{}")
            mapping = json.loads(mapping_raw)
            for name, var in ds.variables.items():
                key = mapping.get(name, name)
                value = var[...]
                if hasattr(value, "filled"):
                    value = value.filled(np.nan)
                value = decode_if_bytes(value)
                original_dtype = getattr(var, "sfincs_jax_original_dtype", "")
                if original_dtype == "bool":
                    value = np.asarray(value, dtype=np.int8).astype(bool)
                out[key] = value
        return out
    raise ValueError(fmt)


__all__ = (
    "ExportFConfig",
    "decode_if_bytes",
    "fortran_logical",
    "fortran_h5_layout",
    "get_namelist_float",
    "get_namelist_int",
    "netcdf_safe_name",
    "output_file_format",
    "read_sfincs_h5",
    "read_sfincs_output_file",
    "to_numpy_for_h5",
    "write_sfincs_h5",
    "write_export_f_state_vectors_to_data",
    "write_sfincs_netcdf",
    "write_sfincs_npz",
    "write_sfincs_output_file",
)
