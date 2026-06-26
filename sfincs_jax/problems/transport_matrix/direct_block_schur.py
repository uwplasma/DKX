"""Direct active block-Schur preconditioner for RHSMode=2/3 transport."""

from __future__ import annotations

from collections.abc import Callable
import os

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioner_caches import (
    _TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_PRECOND_CACHE,
    _TransportFpDirectActiveBlockSchurPrecondCache,
)
from sfincs_jax.problems.profile_response.policies import _hash_numpy_array_for_cache
from sfincs_jax.problems.transport_matrix.active_factor import (
    admit_active_block_schur_factor,
    build_active_block_ordering,
    build_active_block_schur_factor,
    build_active_block_schur_residual_coarse_factor,
    deterministic_probe_matrix,
)
from sfincs_jax.problems.transport_matrix.direct_pmat import _try_build_rhsmode23_fp_direct_active_operator_bundle
from sfincs_jax.v3_system import V3FullSystemOperator
from sfincs_jax.profiling import Timer

__all__ = ["build_transport_fp_direct_active_block_schur_preconditioner"]


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def build_transport_fp_direct_active_block_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    active_indices_np: np.ndarray | None = None,
    emit: Callable[[int, str], None] | None = None,
    fallback_builder: Callable[..., Callable[[jnp.ndarray], jnp.ndarray]],
    transport_precond_cache_key: Callable[[V3FullSystemOperator, str], tuple[object, ...]],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build the bounded direct active block-Schur FP transport preconditioner.

    The module owns setup, cache, admission, residual-coarse rescue, and host
    callback application.  The caller injects the fallback preconditioner and
    cache-key builder because those still live in the driver orchestration
    layer during the active refactor branch.
    """

    if int(op.rhs_mode) not in {2, 3} or op.fblock.fp is None:
        return fallback_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if reduce_full is None or expand_reduced is None or active_indices_np is None:
        return fallback_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

    prefix = "SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR"
    dtype_env = os.environ.get(f"{prefix}_FACTOR_DTYPE", "").strip().lower()
    block_kind_env = os.environ.get(f"{prefix}_BLOCK_KIND", "").strip().lower()
    admission_env = os.environ.get(f"{prefix}_ADMISSION", "").strip().lower()
    coarse_env = os.environ.get(f"{prefix}_RESIDUAL_COARSE", "").strip().lower()
    max_mb = _float_env(f"{prefix}_MAX_MB", 2048.0)
    max_block = _int_env(f"{prefix}_MAX_BLOCK", 64)
    reg = _float_env(f"{prefix}_REG", 1.0e-12)
    tail_max = _int_env(f"{prefix}_TAIL_MAX", 256)
    ell_block = _int_env(f"{prefix}_ELL_BLOCK", 1)
    admission_max_rel = _float_env(f"{prefix}_ADMISSION_MAX_REL", 1.0e-2)
    admission_min_improvement = _float_env(f"{prefix}_ADMISSION_MIN_IMPROVEMENT", 10.0)
    admission_probe_count = _int_env(f"{prefix}_ADMISSION_PROBES", 4)
    coarse_max_cols = _int_env(f"{prefix}_RESIDUAL_COARSE_MAX_COLS", 8)
    coarse_max_mb = _float_env(f"{prefix}_RESIDUAL_COARSE_MAX_MB", 512.0)
    coarse_regularization_rel = _float_env(f"{prefix}_RESIDUAL_COARSE_REGULARIZATION_REL", 1.0e-10)
    coarse_damping = _float_env(f"{prefix}_RESIDUAL_COARSE_DAMPING", 1.0)
    block_kind = block_kind_env if block_kind_env else "zeta_line"
    admission_enabled = admission_env not in {"0", "false", "no", "off"}
    coarse_enabled = coarse_env not in {"0", "false", "no", "off"}
    factor_dtype = np.dtype(np.float32) if dtype_env in {"float32", "fp32", "32"} else np.dtype(np.float64)
    active_indices_use = np.asarray(active_indices_np, dtype=np.int64).reshape((-1,))
    if active_indices_use.size <= 0:
        return fallback_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    active_hash = _hash_numpy_array_for_cache(active_indices_use)
    cache_key = (
        *transport_precond_cache_key(
            op,
            "fp_direct_active_block_schur_"
            f"{block_kind}_{int(ell_block)}_{factor_dtype.name}_{float(reg):.3e}_"
            f"{int(max_block)}_{int(tail_max)}_{int(admission_enabled)}_"
            f"{float(admission_max_rel):.3e}_{float(admission_min_improvement):.3e}_"
            f"{int(admission_probe_count)}_{int(coarse_enabled)}_{int(coarse_max_cols)}_"
            f"{float(coarse_max_mb):.3e}_{float(coarse_regularization_rel):.3e}_"
            f"{float(coarse_damping):.3e}",
        ),
        str(active_hash),
        int(active_indices_use.size),
        int(float(max_mb) * 1.0e6) if float(max_mb) > 0.0 else 0,
    )
    cached = _TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_PRECOND_CACHE.get(cache_key)
    if cached is None:
        try:
            direct_result = _try_build_rhsmode23_fp_direct_active_operator_bundle(
                op=op,
                active_indices=active_indices_use,
                factor_dtype=factor_dtype,
                emit=emit,
            )
            if direct_result is None:
                raise RuntimeError("direct active true operator unavailable")
            operator_bundle, direct_metadata = direct_result
            matrix = operator_bundle.matrix
            if matrix is None:
                raise RuntimeError("direct active true operator has no materialized CSR matrix")
            matrix = matrix.tocsr().astype(factor_dtype, copy=False)
            kinetic_size = int(direct_metadata.get("direct_pmat_kinetic_size", 0))
            tail_size = int(direct_metadata.get("direct_pmat_tail_size", 0))
            active_size = int(matrix.shape[0])
            if kinetic_size <= 0 or kinetic_size > active_size:
                raise RuntimeError("invalid direct active kinetic size")
            if tail_size < 0 or kinetic_size + tail_size != active_size:
                raise RuntimeError("invalid direct active tail size")
            if tail_size > int(tail_max):
                raise RuntimeError(f"tail size {tail_size} exceeds tail_max={int(tail_max)}")

            build_timer = Timer()
            ordering = build_active_block_ordering(
                kinetic_size=int(kinetic_size),
                tail_size=int(tail_size),
                n_theta=int(op.n_theta),
                n_zeta=int(op.n_zeta),
                block_kind=str(block_kind),
                ell_block=int(ell_block),
                max_block_size=int(max_block),
            )
            factor = build_active_block_schur_factor(
                matrix,
                ordering,
                dtype=factor_dtype,
                reg=float(reg),
                max_mb=float(max_mb),
            )
            admission = None
            coarse_metadata: dict[str, object] = {
                "residual_coarse_enabled": bool(coarse_enabled),
                "residual_coarse_accepted": None,
            }
            if bool(admission_enabled):
                probes = deterministic_probe_matrix(
                    active_size=int(active_size),
                    kinetic_size=int(kinetic_size),
                    tail_size=int(tail_size),
                    count=max(1, int(admission_probe_count)),
                )
                admission = admit_active_block_schur_factor(
                    matrix,
                    factor,
                    probes,
                    max_relative_residual=float(admission_max_rel),
                    min_improvement_vs_identity=float(admission_min_improvement),
                )
                if not bool(admission.accepted):
                    if bool(coarse_enabled):
                        try:
                            coarse_factor = build_active_block_schur_residual_coarse_factor(
                                matrix,
                                factor,
                                probes,
                                max_cols=int(coarse_max_cols),
                                regularization_rel=float(coarse_regularization_rel),
                                damping=float(coarse_damping),
                                max_mb=float(coarse_max_mb),
                            )
                            coarse_admission = admit_active_block_schur_factor(
                                matrix,
                                coarse_factor,
                                probes,
                                max_relative_residual=float(admission_max_rel),
                                min_improvement_vs_identity=float(admission_min_improvement),
                            )
                            coarse_metadata = {
                                "residual_coarse_enabled": True,
                                "residual_coarse_accepted": bool(coarse_admission.accepted),
                                "residual_coarse_reason": str(coarse_admission.reason),
                                "residual_coarse_max_relative_residual": float(
                                    coarse_admission.max_relative_residual
                                ),
                                "residual_coarse_median_relative_residual": float(
                                    coarse_admission.median_relative_residual
                                ),
                                "residual_coarse_min_improvement_vs_identity": float(
                                    coarse_admission.min_improvement_vs_identity
                                ),
                                "residual_coarse_probe_count": int(coarse_admission.probe_count),
                                "residual_coarse_max_cols": int(coarse_max_cols),
                                "residual_coarse_max_mb": float(coarse_max_mb),
                                "residual_coarse_regularization_rel": float(coarse_regularization_rel),
                                "residual_coarse_damping": float(coarse_damping),
                            }
                            if bool(coarse_admission.accepted):
                                factor = coarse_factor
                                admission = coarse_admission
                            elif emit is not None:
                                emit(
                                    1,
                                    "solve_v3_transport_matrix_linear_gmres: "
                                    "fp_direct_active_block_schur residual coarse rejected "
                                    f"max_rel={float(coarse_admission.max_relative_residual):.3e} "
                                    f"reason={coarse_admission.reason}",
                                )
                        except Exception as coarse_exc:  # noqa: BLE001
                            coarse_metadata = {
                                "residual_coarse_enabled": True,
                                "residual_coarse_accepted": False,
                                "residual_coarse_error": f"{type(coarse_exc).__name__}: {coarse_exc}",
                                "residual_coarse_max_cols": int(coarse_max_cols),
                                "residual_coarse_max_mb": float(coarse_max_mb),
                                "residual_coarse_regularization_rel": float(coarse_regularization_rel),
                                "residual_coarse_damping": float(coarse_damping),
                            }
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_transport_matrix_linear_gmres: "
                                    "fp_direct_active_block_schur residual coarse unavailable "
                                    f"({type(coarse_exc).__name__}: {coarse_exc})",
                                )
                    if not bool(admission.accepted):
                        coarse_suffix = ""
                        if coarse_metadata.get("residual_coarse_accepted") is False:
                            coarse_suffix = (
                                ", residual_coarse="
                                f"{coarse_metadata.get('residual_coarse_reason', coarse_metadata.get('residual_coarse_error', 'rejected'))}"
                            )
                        raise RuntimeError(
                            "admission rejected direct active block-Schur "
                            f"(block_kind={factor.ordering.block_kind}, blocks={len(factor.ordering.blocks)}, "
                            f"reason={admission.reason}, max_rel={admission.max_relative_residual:.3e}, "
                            f"min_improvement={admission.min_improvement_vs_identity:.3e}{coarse_suffix})"
                        )
            else:
                coarse_metadata = {
                    "residual_coarse_enabled": bool(coarse_enabled),
                    "residual_coarse_accepted": None,
                    "residual_coarse_reason": "admission_disabled",
                }

            factor_metadata = dict(factor.metadata)
            metadata = {
                "kind": "fp_direct_active_block_schur",
                "factor_dtype": str(factor_dtype.name),
                "axis": str(factor_metadata.get("block_kind", block_kind)),
                "block_kind": str(factor_metadata.get("block_kind", block_kind)),
                "block_size": int(factor_metadata.get("block_size_max", 0)),
                "block_count": int(factor_metadata.get("block_count", 0)),
                "kinetic_size": int(kinetic_size),
                "tail_size": int(tail_size),
                "reg": float(reg),
                "matrix_nbytes_estimate": int(factor_metadata.get("matrix_nbytes_estimate", 0)),
                "block_inverse_nbytes_estimate": int(factor_metadata.get("inverse_nbytes_estimate", 0)),
                "tail_dense_nbytes_estimate": int(factor_metadata.get("tail_nbytes_estimate", 0)),
                "total_nbytes_estimate": int(factor_metadata.get("total_nbytes_estimate", 0)),
                "setup_s": float(build_timer.elapsed_s()),
                "schur_reason": "dense_schur" if int(tail_size) > 0 else "none",
                "admission_enabled": bool(admission_enabled),
                "admission_accepted": None if admission is None else bool(admission.accepted),
                "admission_reason": None if admission is None else str(admission.reason),
                "admission_max_relative_residual": None
                if admission is None
                else float(admission.max_relative_residual),
                "admission_median_relative_residual": None
                if admission is None
                else float(admission.median_relative_residual),
                "admission_min_improvement_vs_identity": None
                if admission is None
                else float(admission.min_improvement_vs_identity),
                "admission_probe_count": None if admission is None else int(admission.probe_count),
                "residual_coarse_enabled": bool(coarse_enabled),
            }
            metadata.update(direct_metadata)
            metadata.update(coarse_metadata)
            cached = _TransportFpDirectActiveBlockSchurPrecondCache(
                block_inverse=(),
                block_size=int(factor.ordering.block_size_max),
                kinetic_size=int(kinetic_size),
                tail_size=int(tail_size),
                c_tail=factor.c_tail,
                mb_tail=None if factor.mb_tail is None else np.asarray(factor.mb_tail, dtype=factor_dtype),
                schur_inverse=None if factor.schur_inverse is None else np.asarray(factor.schur_inverse, dtype=factor_dtype),
                metadata=metadata,
                factor=factor,
            )
            _TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_PRECOND_CACHE[cache_key] = cached
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: fp_direct_active_block_schur selected "
                    f"active={active_size} kinetic={kinetic_size} tail={tail_size} "
                    f"blocks={int(metadata['block_count'])}x<= {int(metadata['block_size'])} "
                    f"setup_s={float(metadata['setup_s']):.3f} "
                    f"est_mb={float(metadata['total_nbytes_estimate']) / 1.0e6:.3f}",
                )
        except Exception as exc:  # noqa: BLE001
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: fp_direct_active_block_schur unavailable; "
                    f"using sxblock ({type(exc).__name__}: {exc})",
                )
            return fallback_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

    block_inverse = cached.block_inverse
    block_size = int(cached.block_size)
    kinetic_size = int(cached.kinetic_size)
    tail_size = int(cached.tail_size)
    n_blocks = int(kinetic_size // block_size)
    c_tail = cached.c_tail
    mb_tail = cached.mb_tail
    schur_inverse = cached.schur_inverse
    factor_obj = getattr(cached, "factor", None)
    if factor_obj is not None:
        factor_dtype_use = np.dtype(getattr(factor_obj, "dtype", np.float64))
    else:
        factor_dtype_use = np.dtype(getattr(block_inverse, "dtype", np.float64))
    active_size_use = int(kinetic_size + tail_size)

    def _apply_host(rhs_host: np.ndarray) -> np.ndarray:
        if factor_obj is not None:
            return np.asarray(factor_obj.apply(rhs_host), dtype=np.float64)
        rhs_np = np.asarray(rhs_host, dtype=factor_dtype_use).reshape((active_size_use,))
        rhs_k = rhs_np[:kinetic_size]
        rhs_blocks = rhs_k.reshape((n_blocks, block_size, 1))
        y_k = np.einsum("bij,bjk->bik", block_inverse, rhs_blocks, optimize=True).reshape((kinetic_size,))
        if tail_size > 0 and c_tail is not None and mb_tail is not None and schur_inverse is not None:
            rhs_t = rhs_np[kinetic_size:]
            tail_residual = np.asarray(rhs_t - c_tail @ y_k, dtype=factor_dtype_use).reshape((tail_size,))
            y_t = np.asarray(schur_inverse @ tail_residual, dtype=factor_dtype_use).reshape((tail_size,))
            y_k = np.asarray(y_k - mb_tail @ y_t, dtype=factor_dtype_use).reshape((kinetic_size,))
            out = np.concatenate([y_k, y_t], axis=0)
        else:
            out = np.concatenate([y_k, rhs_np[kinetic_size:]], axis=0)
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=np.float64)

    def _apply_reduced(v: jnp.ndarray) -> jnp.ndarray:
        v = jnp.asarray(v, dtype=jnp.float64)
        return jax.pure_callback(
            _apply_host,
            jax.ShapeDtypeStruct((active_size_use,), jnp.float64),
            v,
        )

    try:
        setattr(_apply_reduced, "_sfincs_jax_transport_fp_direct_active_block_schur_metadata", dict(cached.metadata))
    except Exception:
        pass
    return _apply_reduced
