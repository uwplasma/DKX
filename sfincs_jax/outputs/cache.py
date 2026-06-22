"""Output geometry-cache keys and persistence helpers.

The public output writer builds several geometry-derived datasets before a
solve result is available. These helpers keep the cache gate, cache key, and
disk persistence independent of the high-level ``io.py`` writer.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any
import hashlib
import os

import numpy as np

from ..namelist import Namelist
from ..v3 import V3Grids

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


__all__ = (
    "OUTPUT_CACHE_FIELDS",
    "OUTPUT_GEOM_CACHE",
    "OUTPUT_GEOM_CACHE_VERSION",
    "equilibrium_cache_identity",
    "file_content_identity",
    "group_subset_key",
    "hashable_value",
    "load_output_cache",
    "output_cache_dir",
    "output_cache_enabled",
    "output_cache_path",
    "output_geom_cache_key",
    "save_output_cache",
)
