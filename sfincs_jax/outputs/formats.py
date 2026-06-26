"""Flat SFINCS output readers and writers.

This module owns the on-disk format boundary for small and moderate
``sfincsOutput`` payloads. Solver and diagnostic code should construct a plain
dataset dictionary; this layer serializes it to HDF5, NetCDF4, or NPZ while
preserving the SFINCS Fortran readback layout convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

import h5py
import numpy as np

from ..solvers.diagnostics import SolverTrace, write_solver_trace_h5


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
    "decode_if_bytes",
    "fortran_h5_layout",
    "netcdf_safe_name",
    "output_file_format",
    "read_sfincs_h5",
    "read_sfincs_output_file",
    "to_numpy_for_h5",
    "write_sfincs_h5",
    "write_sfincs_netcdf",
    "write_sfincs_npz",
    "write_sfincs_output_file",
)
