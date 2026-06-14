"""Environment and factor-kind policy for explicit sparse host factors."""

from __future__ import annotations

from collections.abc import Mapping
import os


EnvMapping = Mapping[str, str]


_FACTOR_KIND_ALIASES = {
    "jacobi": "jacobi",
    "diagonal": "jacobi",
    "diag": "jacobi",
    "none": "jacobi",
    "symbolic_block_schur_lu": "symbolic_block_schur_lu",
    "block_schur_lu": "symbolic_block_schur_lu",
    "native_block_schur_lu": "symbolic_block_schur_lu",
    "symbolic_schur_lu": "symbolic_block_schur_lu",
    "symbolic_frontal_schur_lu": "symbolic_frontal_schur_lu",
    "frontal_schur_lu": "symbolic_frontal_schur_lu",
    "native_frontal_schur_lu": "symbolic_frontal_schur_lu",
    "multifrontal_schur_lu": "symbolic_frontal_schur_lu",
    "symbolic_blr_frontal_schur_lu": "symbolic_blr_frontal_schur_lu",
    "blr_frontal_schur_lu": "symbolic_blr_frontal_schur_lu",
    "native_blr_frontal_schur_lu": "symbolic_blr_frontal_schur_lu",
    "compressed_frontal_schur_lu": "symbolic_blr_frontal_schur_lu",
    "symbolic_nd_frontal_schur_lu": "symbolic_nd_frontal_schur_lu",
    "nd_frontal_schur_lu": "symbolic_nd_frontal_schur_lu",
    "nested_dissection_frontal_schur_lu": "symbolic_nd_frontal_schur_lu",
    "native_nd_frontal_schur_lu": "symbolic_nd_frontal_schur_lu",
    "multilevel_frontal_schur_lu": "symbolic_nd_frontal_schur_lu",
    "symbolic_superblock_lu": "symbolic_superblock_lu",
    "superblock_lu": "symbolic_superblock_lu",
    "native_superblock_lu": "symbolic_superblock_lu",
    "block_edge_lu": "symbolic_superblock_lu",
    "symbolic_block_lu_coarse": "symbolic_block_lu_coarse",
    "block_lu_coarse": "symbolic_block_lu_coarse",
    "native_block_lu_coarse": "symbolic_block_lu_coarse",
    "symbolic_lu_coarse": "symbolic_block_lu_coarse",
    "symbolic_block_lu": "symbolic_block_lu",
    "block_lu": "symbolic_block_lu",
    "native_block_lu": "symbolic_block_lu",
    "symbolic_lu": "symbolic_block_lu",
    "ilu": "ilu",
    "spilu": "ilu",
    "lu": "lu",
    "splu": "lu",
}


def parse_explicit_sparse_int(value: str, default: int, *, minimum: int = 0) -> int:
    """Parse an integer explicit-sparse option with fail-closed bounds."""

    try:
        parsed = int(value) if value else int(default)
    except ValueError:
        parsed = int(default)
    return max(int(minimum), int(parsed))


def parse_explicit_sparse_float(value: str, default: float, *, minimum: float = 0.0) -> float:
    """Parse a floating-point explicit-sparse option with fail-closed bounds."""

    try:
        parsed = float(value) if value else float(default)
    except ValueError:
        parsed = float(default)
    return max(float(minimum), float(parsed))


def parse_explicit_sparse_bool(value: str, default: bool) -> bool:
    """Parse Fortran/Python-style boolean explicit-sparse options."""

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on", ".true.", ".t."}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off", ".false.", ".f."}:
        return False
    return bool(default)


def canonical_explicit_sparse_factor_kind(kind: str | None, *, default: str = "lu") -> str:
    """Normalize explicit-sparse factor aliases to canonical factor kinds."""

    kind_l = str(kind or "").strip().lower()
    if kind_l in _FACTOR_KIND_ALIASES:
        return _FACTOR_KIND_ALIASES[kind_l]
    default_l = str(default or "").strip().lower()
    return _FACTOR_KIND_ALIASES.get(default_l, "lu")


def explicit_sparse_factor_kind_from_env(
    default_factor_kind: str,
    *,
    env: EnvMapping | None = None,
) -> str:
    """Resolve the explicit sparse factor kind from env, then default aliases."""

    env_map = os.environ if env is None else env
    override = str(env_map.get("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "")).strip().lower()
    if override:
        return canonical_explicit_sparse_factor_kind(override, default=default_factor_kind)
    return canonical_explicit_sparse_factor_kind(default_factor_kind, default="lu")


def explicit_sparse_monolithic_guard_enabled(
    default_enabled: bool,
    *,
    env: EnvMapping | None = None,
) -> bool:
    """Resolve whether monolithic LU/ILU preflight guards are enabled."""

    env_map = os.environ if env is None else env
    return parse_explicit_sparse_bool(
        str(env_map.get("SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_GUARD", "")).strip(),
        bool(default_enabled),
    )


def explicit_sparse_monolithic_max_size(
    factor_kind: str,
    *,
    env: EnvMapping | None = None,
    default: int = 250_000,
) -> int:
    """Resolve factor-specific monolithic LU/ILU maximum active size."""

    env_map = os.environ if env is None else env
    max_n_name = (
        "SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_LU_MAX_SIZE"
        if str(factor_kind).strip().lower() == "lu"
        else "SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_ILU_MAX_SIZE"
    )
    max_n_env = str(env_map.get(max_n_name, "")).strip()
    max_n_fallback_env = str(env_map.get("SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_MAX_SIZE", "")).strip()
    return parse_explicit_sparse_int(max_n_env or max_n_fallback_env, int(default), minimum=0)


__all__ = [
    "canonical_explicit_sparse_factor_kind",
    "explicit_sparse_factor_kind_from_env",
    "explicit_sparse_monolithic_guard_enabled",
    "explicit_sparse_monolithic_max_size",
    "parse_explicit_sparse_bool",
    "parse_explicit_sparse_float",
    "parse_explicit_sparse_int",
]
