from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental import sparse as jsparse

from .explicit_sparse import estimate_csr_nbytes

ReferenceInputKind = Literal["jax", "numpy"]


@dataclass(frozen=True)
class DeviceCSRMetadata:
    """Host/device storage summary for a materialized CSR operator."""

    shape: tuple[int, int]
    nnz: int
    data_dtype: str
    index_dtype: str
    csr_nbytes: int
    max_csr_nbytes: int | None
    source: str
    row_indices_nbytes: int = 0
    active_size: int | None = None
    source_shape: tuple[int, int] | None = None
    source_nnz: int | None = None
    active_mapping_nbytes: int = 0
    drop_tol: float = 0.0
    requested_device: str | None = None
    default_backend: str = ""
    available_platforms: tuple[str, ...] = ()
    array_devices: tuple[str, ...] = ()
    array_platforms: tuple[str, ...] = ()
    all_arrays_same_device: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": self.shape,
            "nnz": int(self.nnz),
            "data_dtype": self.data_dtype,
            "index_dtype": self.index_dtype,
            "csr_nbytes": int(self.csr_nbytes),
            "max_csr_nbytes": self.max_csr_nbytes,
            "source": self.source,
            "row_indices_nbytes": int(self.row_indices_nbytes),
            "active_size": self.active_size,
            "source_shape": self.source_shape,
            "source_nnz": self.source_nnz,
            "active_mapping_nbytes": int(self.active_mapping_nbytes),
            "drop_tol": float(self.drop_tol),
            "requested_device": self.requested_device,
            "default_backend": self.default_backend,
            "available_platforms": tuple(self.available_platforms),
            "array_devices": tuple(self.array_devices),
            "array_platforms": tuple(self.array_platforms),
            "all_arrays_same_device": bool(self.all_arrays_same_device),
        }


@dataclass(frozen=True)
class DeviceCSR:
    """JAX-device CSR arrays plus a pure JAX matvec closure."""

    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    shape: tuple[int, int]
    metadata: DeviceCSRMetadata
    active_indices: jax.Array | None = None
    row_indices: jax.Array | None = None

    def arrays(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        return self.data, self.indices, self.indptr

    @property
    def nnz(self) -> int:
        return int(self.metadata.nnz)

    @property
    def nbytes_estimate(self) -> int:
        return int(self.metadata.csr_nbytes)

    def matvec(self, x: jax.Array) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=self.data.dtype)
        if x_arr.ndim != 1:
            raise ValueError(f"matvec expects a 1D vector, got shape {x_arr.shape}")
        if int(x_arr.shape[0]) != int(self.shape[1]):
            raise ValueError(f"matvec vector length {x_arr.shape[0]} does not match operator columns {self.shape[1]}")
        csr = jsparse.CSR((self.data, self.indices, self.indptr), shape=self.shape)
        return jsparse.csr_matvec(csr, x_arr)

    def as_matvec(self) -> Callable[[jax.Array], jax.Array]:
        return self.matvec

    def jitted_matvec(self) -> Callable[[jax.Array], jax.Array]:
        return jax.jit(self.matvec)


@dataclass(frozen=True)
class MatvecValidationResult:
    samples: int
    passed: bool
    max_abs_error: float
    max_rel_error: float
    rel_errors: tuple[float, ...]
    rtol: float
    atol: float
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": int(self.samples),
            "passed": bool(self.passed),
            "max_abs_error": float(self.max_abs_error),
            "max_rel_error": float(self.max_rel_error),
            "rel_errors": tuple(float(v) for v in self.rel_errors),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "seed": int(self.seed),
        }


def estimate_device_csr_nbytes(
    shape: tuple[int, int],
    nnz: int,
    *,
    data_dtype: Any = np.float64,
    index_dtype: Any = np.int32,
) -> int:
    return int(
        estimate_csr_nbytes(
            tuple(int(v) for v in shape),
            int(nnz),
            data_dtype=np.dtype(data_dtype),
            index_dtype=np.dtype(index_dtype),
        )
    )


def _resolve_max_csr_nbytes(*, max_csr_nbytes: int | None, max_csr_mb: float | None) -> int | None:
    limits: list[int] = []
    if max_csr_nbytes is not None:
        limits.append(max(0, int(max_csr_nbytes)))
    if max_csr_mb is not None:
        limits.append(int(max(0.0, float(max_csr_mb)) * 1.0e6))
    return min(limits) if limits else None


def _device_string(device: Any) -> str:
    if device is None:
        return ""
    return str(device)


def _device_platform(device: Any) -> str:
    return str(getattr(device, "platform", "") or "")


def _array_devices(array: Any) -> tuple[Any, ...]:
    if array is None:
        return ()
    devices_attr = getattr(array, "devices", None)
    if callable(devices_attr):
        try:
            return tuple(sorted(devices_attr(), key=str))
        except TypeError:
            return ()
    device_attr = getattr(array, "device", None)
    if callable(device_attr):
        try:
            return (device_attr(),)
        except TypeError:
            return ()
    if device_attr is not None:
        return (device_attr,)
    return ()


def _placement_metadata(
    arrays: tuple[Any, ...],
    *,
    requested_device: jax.Device | None,
) -> dict[str, object]:
    devices: list[Any] = []
    for array in arrays:
        devices.extend(_array_devices(array))
    unique_devices = tuple(sorted({_device_string(device) for device in devices if device is not None}))
    platforms = tuple(sorted({_device_platform(device) for device in devices if device is not None and _device_platform(device)}))
    try:
        default_backend = str(jax.default_backend())
    except RuntimeError:
        default_backend = ""
    try:
        available_platforms = tuple(sorted({str(device.platform) for device in jax.devices()}))
    except RuntimeError:
        available_platforms = ()
    return {
        "requested_device": None if requested_device is None else _device_string(requested_device),
        "default_backend": default_backend,
        "available_platforms": available_platforms,
        "array_devices": unique_devices,
        "array_platforms": platforms,
        "all_arrays_same_device": bool(unique_devices) and len(unique_devices) == 1,
    }


def _check_csr_budget(metadata: DeviceCSRMetadata) -> None:
    if metadata.max_csr_nbytes is not None and int(metadata.csr_nbytes) > int(metadata.max_csr_nbytes):
        raise MemoryError(
            "device CSR operator exceeds memory budget "
            f"({metadata.csr_nbytes / 1.0e6:.3g} MB > {metadata.max_csr_nbytes / 1.0e6:.3g} MB)"
        )


def _normalize_active_indices(active_indices: Any, *, size: int) -> np.ndarray:
    active = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if active.size == 0:
        return active
    if np.any(active < 0) or np.any(active >= int(size)):
        raise ValueError("active_indices contains entries outside the operator shape")
    if np.unique(active).size != active.size:
        raise ValueError("active_indices must not contain duplicates")
    return active


def _index_dtype_checked(*, shape: tuple[int, int], nnz: int, index_dtype: Any) -> np.dtype:
    dtype = np.dtype(index_dtype)
    if not np.issubdtype(dtype, np.signedinteger):
        raise TypeError("index_dtype must be a signed integer dtype")
    info = np.iinfo(dtype)
    max_required = max(int(shape[0]), int(shape[1]), int(nnz))
    if max_required > int(info.max):
        raise OverflowError(f"operator shape/nnz require indices larger than {dtype.name}")
    return dtype


def _copy_csr_with_drop(csr: sp.spmatrix, *, data_dtype: np.dtype, drop_tol: float) -> sp.csr_matrix:
    out = csr.tocsr(copy=True).astype(data_dtype, copy=False)
    out.sum_duplicates()
    if float(drop_tol) > 0.0 and out.nnz:
        out.data[np.abs(out.data) <= float(drop_tol)] = 0
    out.eliminate_zeros()
    out.sort_indices()
    return out


def _slice_csr_active(
    csr: sp.csr_matrix,
    active_indices: np.ndarray,
    *,
    data_dtype: np.dtype,
    index_dtype: np.dtype,
    drop_tol: float,
) -> tuple[sp.csr_matrix, int]:
    if csr.shape[0] != csr.shape[1]:
        raise ValueError("active_indices slicing requires a square operator")
    active = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    active_size = int(active.size)
    full_to_active = np.full((int(csr.shape[1]),), -1, dtype=index_dtype)
    full_to_active[active] = np.arange(active_size, dtype=index_dtype)

    indptr = np.empty((active_size + 1,), dtype=index_dtype)
    indptr[0] = 0
    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    nnz = 0
    for out_row, row in enumerate(active):
        start = int(csr.indptr[int(row)])
        end = int(csr.indptr[int(row) + 1])
        cols = np.asarray(csr.indices[start:end], dtype=np.int64)
        values = np.asarray(csr.data[start:end], dtype=data_dtype)
        mapped = full_to_active[cols]
        keep = mapped >= 0
        if float(drop_tol) > 0.0:
            keep = keep & (np.abs(values) > float(drop_tol))
        if np.any(keep):
            values_kept = np.asarray(values[keep], dtype=data_dtype)
            cols_kept = np.asarray(mapped[keep], dtype=index_dtype)
            data_parts.append(values_kept)
            index_parts.append(cols_kept)
            nnz += int(values_kept.size)
        indptr[out_row + 1] = nnz

    if data_parts:
        data = np.concatenate(data_parts).astype(data_dtype, copy=False)
        indices = np.concatenate(index_parts).astype(index_dtype, copy=False)
    else:
        data = np.asarray([], dtype=data_dtype)
        indices = np.asarray([], dtype=index_dtype)
    sliced = sp.csr_matrix((data, indices, indptr), shape=(active_size, active_size), dtype=data_dtype)
    sliced.sum_duplicates()
    sliced.eliminate_zeros()
    sliced.sort_indices()
    return sliced, int(full_to_active.nbytes)


def _extract_materialized_matrix(operator: Any) -> Any:
    matrix = operator
    if not sp.issparse(matrix) and hasattr(operator, "matrix"):
        matrix = getattr(operator, "matrix")
    if matrix is None:
        raise ValueError("operator does not contain a materialized matrix")
    return matrix


def materialized_operator_to_csr(
    operator: Any,
    *,
    allow_dense: bool = False,
    data_dtype: Any | None = np.float64,
    drop_tol: float = 0.0,
) -> sp.csr_matrix:
    """Extract a host CSR matrix from a SciPy sparse matrix or materialized bundle."""

    matrix = _extract_materialized_matrix(operator)
    dtype = None if data_dtype is None else np.dtype(data_dtype)
    if sp.issparse(matrix):
        return _copy_csr_with_drop(
            matrix,
            data_dtype=np.dtype(matrix.dtype if dtype is None else dtype),
            drop_tol=drop_tol,
        )

    if not allow_dense:
        raise TypeError(
            "expected a SciPy sparse matrix or sparse materialized operator; "
            "pass allow_dense=True to convert dense input"
        )
    arr = np.asarray(jax.device_get(matrix), dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {arr.shape}")
    return _copy_csr_with_drop(sp.csr_matrix(arr), data_dtype=np.dtype(arr.dtype), drop_tol=drop_tol)


def device_csr_from_scipy_csr(
    matrix: sp.spmatrix,
    *,
    active_indices: Any | None = None,
    data_dtype: Any | None = np.float64,
    index_dtype: Any = np.int32,
    max_csr_nbytes: int | None = None,
    max_csr_mb: float | None = None,
    drop_tol: float = 0.0,
    device: jax.Device | None = None,
    source: str = "scipy_csr",
) -> DeviceCSR:
    """Place a SciPy sparse matrix on device as bounded CSR arrays."""

    if not sp.issparse(matrix):
        raise TypeError("matrix must be a SciPy sparse matrix")
    source_shape = tuple(int(v) for v in matrix.shape)
    source_nnz = int(matrix.nnz)
    dtype = np.dtype(matrix.dtype if data_dtype is None else data_dtype)
    csr_source = matrix.tocsr(copy=False)
    active_mapping_nbytes = 0
    if active_indices is not None:
        active = _normalize_active_indices(active_indices, size=int(source_shape[0]))
        _index_dtype_checked(shape=(int(active.size), int(active.size)), nnz=int(active.size), index_dtype=index_dtype)
        csr, active_mapping_nbytes = _slice_csr_active(
            csr_source,
            active,
            data_dtype=dtype,
            index_dtype=np.dtype(index_dtype),
            drop_tol=float(drop_tol),
        )
        active_size: int | None = int(active.size)
    else:
        csr = _copy_csr_with_drop(csr_source, data_dtype=dtype, drop_tol=float(drop_tol))
        active = None
        active_size = None

    index_dtype_np = _index_dtype_checked(shape=tuple(csr.shape), nnz=int(csr.nnz), index_dtype=index_dtype)
    csr_nbytes = estimate_device_csr_nbytes(
        tuple(csr.shape),
        int(csr.nnz),
        data_dtype=dtype,
        index_dtype=index_dtype_np,
    )
    max_nbytes = _resolve_max_csr_nbytes(max_csr_nbytes=max_csr_nbytes, max_csr_mb=max_csr_mb)
    metadata = DeviceCSRMetadata(
        shape=tuple(int(v) for v in csr.shape),
        nnz=int(csr.nnz),
        data_dtype=np.dtype(dtype).name,
        index_dtype=index_dtype_np.name,
        csr_nbytes=int(csr_nbytes),
        max_csr_nbytes=max_nbytes,
        source=str(source),
        row_indices_nbytes=0,
        active_size=active_size,
        source_shape=source_shape,
        source_nnz=source_nnz,
        active_mapping_nbytes=int(active_mapping_nbytes),
        drop_tol=float(drop_tol),
    )
    _check_csr_budget(metadata)

    data = jax.device_put(np.asarray(csr.data, dtype=dtype), device=device)
    indices = jax.device_put(np.asarray(csr.indices, dtype=index_dtype_np), device=device)
    indptr = jax.device_put(np.asarray(csr.indptr, dtype=index_dtype_np), device=device)
    active_jax = None if active is None else jax.device_put(np.asarray(active, dtype=index_dtype_np), device=device)
    metadata = replace(
        metadata,
        **_placement_metadata(
            (data, indices, indptr, active_jax),
            requested_device=device,
        ),
    )
    return DeviceCSR(
        data=jnp.asarray(data),
        indices=jnp.asarray(indices),
        indptr=jnp.asarray(indptr),
        shape=tuple(int(v) for v in csr.shape),
        metadata=metadata,
        active_indices=None if active_jax is None else jnp.asarray(active_jax),
    )


def device_csr_from_operator(
    operator: Any,
    *,
    active_indices: Any | None = None,
    allow_dense: bool = False,
    data_dtype: Any | None = np.float64,
    index_dtype: Any = np.int32,
    max_csr_nbytes: int | None = None,
    max_csr_mb: float | None = None,
    drop_tol: float = 0.0,
    device: jax.Device | None = None,
    source: str = "materialized_operator",
) -> DeviceCSR:
    matrix = _extract_materialized_matrix(operator)
    if not sp.issparse(matrix):
        if not allow_dense:
            raise TypeError(
                "expected a SciPy sparse matrix or sparse materialized operator; "
                "pass allow_dense=True to convert dense input"
            )
        dtype = None if data_dtype is None else np.dtype(data_dtype)
        arr = np.asarray(jax.device_get(matrix), dtype=dtype)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2D matrix, got shape {arr.shape}")
        matrix = sp.csr_matrix(arr)
    return device_csr_from_scipy_csr(
        matrix,
        active_indices=active_indices,
        data_dtype=data_dtype,
        index_dtype=index_dtype,
        max_csr_nbytes=max_csr_nbytes,
        max_csr_mb=max_csr_mb,
        drop_tol=float(drop_tol),
        device=device,
        source=source,
    )


def device_csr_from_matrix(
    matrix: Any,
    *,
    active_indices: Any | None = None,
    dtype: Any | None = np.float64,
    index_dtype: Any = np.int32,
    max_nbytes: int | None = None,
    max_mb: float | None = None,
    drop_tol: float = 0.0,
    device: jax.Device | None = None,
    source: str = "materialized_matrix",
) -> DeviceCSR:
    return device_csr_from_operator(
        matrix,
        active_indices=active_indices,
        allow_dense=True,
        data_dtype=dtype,
        index_dtype=index_dtype,
        max_csr_nbytes=max_nbytes,
        max_csr_mb=max_mb,
        drop_tol=drop_tol,
        device=device,
        source=source,
    )


def validate_device_matvec(
    device_operator: DeviceCSR,
    reference_matvec: Callable[[Any], Any],
    *,
    samples: int = 3,
    probes: Any | None = None,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    seed: int = 1729,
    reference_input: ReferenceInputKind = "jax",
) -> MatvecValidationResult:
    """Compare a device CSR matvec against an existing matrix-free matvec."""

    if reference_input not in {"jax", "numpy"}:
        raise ValueError("reference_input must be 'jax' or 'numpy'")
    n_cols = int(device_operator.shape[1])
    dtype = device_operator.data.dtype
    probe_arrays: list[np.ndarray] = []
    if probes is not None:
        probe_np = np.asarray(jax.device_get(probes), dtype=np.float64)
        if probe_np.ndim == 1:
            probe_np = probe_np.reshape((1, -1))
        if probe_np.ndim != 2 or int(probe_np.shape[1]) != n_cols:
            raise ValueError(f"probes must have shape (samples, {n_cols}) or ({n_cols},), got {probe_np.shape}")
        probe_arrays.extend(np.asarray(row, dtype=np.float64) for row in probe_np)

    rng = np.random.default_rng(int(seed))
    for _ in range(max(0, int(samples))):
        probe = rng.standard_normal(n_cols).astype(np.float64)
        norm = float(np.linalg.norm(probe))
        if np.isfinite(norm) and norm > 0.0:
            probe /= norm
        probe_arrays.append(probe)

    rel_errors: list[float] = []
    max_abs = 0.0
    passed = True
    for probe in probe_arrays:
        probe_for_device = jnp.asarray(probe, dtype=dtype)
        got = np.asarray(jax.device_get(device_operator.matvec(probe_for_device)))
        ref_arg = np.asarray(probe, dtype=np.dtype(dtype)) if reference_input == "numpy" else probe_for_device
        ref = np.asarray(jax.device_get(reference_matvec(ref_arg)))
        if got.shape != ref.shape:
            raise ValueError(f"reference matvec returned shape {ref.shape}; expected {got.shape}")
        diff = got - ref
        abs_err = float(np.max(np.abs(diff))) if diff.size else 0.0
        ref_norm = float(np.linalg.norm(ref))
        rel_err = float(np.linalg.norm(diff) / max(ref_norm, 1.0e-300))
        max_abs = max(max_abs, abs_err)
        rel_errors.append(rel_err)
        passed = bool(passed and np.allclose(got, ref, rtol=float(rtol), atol=float(atol)))

    return MatvecValidationResult(
        samples=int(len(probe_arrays)),
        passed=bool(passed),
        max_abs_error=float(max_abs),
        max_rel_error=max(rel_errors, default=0.0),
        rel_errors=tuple(float(v) for v in rel_errors),
        rtol=float(rtol),
        atol=float(atol),
        seed=int(seed),
    )


def assert_device_matvec_matches(
    device_operator: DeviceCSR,
    reference_matvec: Callable[[Any], Any],
    **kwargs: Any,
) -> MatvecValidationResult:
    result = validate_device_matvec(device_operator, reference_matvec, **kwargs)
    if not result.passed:
        raise AssertionError(
            "device CSR matvec validation failed "
            f"max_abs_error={result.max_abs_error:.3e} max_rel_error={result.max_rel_error:.3e}"
        )
    return result


def validate_device_csr_matvec(
    device_operator: DeviceCSR,
    reference_matvec: Callable[[Any], Any],
    *,
    samples: int = 3,
    probes: Any | None = None,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    seed: int = 1729,
    reference_input: ReferenceInputKind = "numpy",
) -> tuple[float, ...]:
    result = assert_device_matvec_matches(
        device_operator,
        reference_matvec,
        samples=samples,
        probes=probes,
        rtol=rtol,
        atol=atol,
        seed=seed,
        reference_input=reference_input,
    )
    return result.rel_errors


__all__ = [
    "DeviceCSR",
    "DeviceCSRMetadata",
    "MatvecValidationResult",
    "assert_device_matvec_matches",
    "device_csr_from_matrix",
    "device_csr_from_operator",
    "device_csr_from_scipy_csr",
    "estimate_device_csr_nbytes",
    "materialized_operator_to_csr",
    "validate_device_csr_matvec",
    "validate_device_matvec",
]
