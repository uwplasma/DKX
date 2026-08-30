from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _strip_quotes(s: str) -> str:
    return s.strip().strip('"').strip("'")


@dataclass(frozen=True)
class ResolveResult:
    path: Path
    tried: tuple[Path, ...]


def resolve_existing_path(
    path: str | Path,
    *,
    base_dir: Path | None = None,
    env_search_var: str = "DKX_EQUILIBRIA_DIRS",
    extra_search_dirs: tuple[Path, ...] = (),
) -> ResolveResult:
    """Resolve a possibly-relative path string to an existing file.

    Resolution order:
      1) Absolute path as given.
      2) Relative to `base_dir` (if provided).
      3) Relative to `Path.cwd()`.
      4) Directories listed in `env_search_var` (OS pathsep-separated). For each directory `d`,
         relative paths try both `d / p` and `d / p.name`; missing absolute paths try
         `d / p.name` so copied decks with stale machine-local prefixes can be redirected.
      5) Any `extra_search_dirs` (same basename fallback for missing absolute paths).

    Returns the resolved path and a record of all attempted candidate paths.
    """
    if isinstance(path, Path):
        raw = str(path)
    else:
        raw = str(path)
    p = Path(_strip_quotes(raw))

    tried: list[Path] = []

    def _try(candidate: Path) -> Path | None:
        c = candidate.expanduser()
        tried.append(c)
        if c.exists():
            return c
        return None

    if p.is_absolute():
        found = _try(p)
        if found is not None:
            return ResolveResult(path=found, tried=tuple(tried))
    else:
        if base_dir is not None:
            found = _try((base_dir / p).resolve())
            if found is not None:
                return ResolveResult(path=found, tried=tuple(tried))
        found = _try((Path.cwd() / p).resolve())
        if found is not None:
            return ResolveResult(path=found, tried=tuple(tried))

    env_dirs = os.environ.get(env_search_var, "")
    if env_dirs:
        for d in env_dirs.split(os.pathsep):
            if not d:
                continue
            root = Path(_strip_quotes(d)).expanduser()
            if not p.is_absolute():
                found = _try((root / p).resolve())
                if found is not None:
                    return ResolveResult(path=found, tried=tuple(tried))
            found = _try((root / p.name).resolve())
            if found is not None:
                return ResolveResult(path=found, tried=tuple(tried))

    for root in extra_search_dirs:
        if not p.is_absolute():
            found = _try((root / p).resolve())
            if found is not None:
                return ResolveResult(path=found, tried=tuple(tried))
        found = _try((root / p.name).resolve())
        if found is not None:
            return ResolveResult(path=found, tried=tuple(tried))

    # Public examples reference several multi-megabyte equilibrium fixtures by
    # basename. These fixtures live in a release asset instead of the git tree,
    # so resolve them lazily into the user cache when requested.
    try:
        from .validation.data_fetch import resolve_external_equilibrium

        found = resolve_external_equilibrium(p)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(f"Unable to resolve existing path for {raw!r}. Tried: {tried}") from exc
    if found is not None:
        tried.append(found)
        return ResolveResult(path=found, tried=tuple(tried))

    raise FileNotFoundError(f"Unable to resolve existing path for {raw!r}. Tried: {tried}")


#: Marker that identifies a DKX source checkout. Chosen because it is the one
#: file that must exist at the root of a checkout and never inside the
#: installed package.
_CHECKOUT_MARKER = "pyproject.toml"


def repository_root() -> Path | None:
    """Return the DKX checkout that contains this package, or None.

    Six call sites used to spell this as ``Path(__file__).resolve().parents[N]``
    with N hand-counted from the module's depth. Moving the package under
    ``src/`` made every one of them off by one, silently: they still returned a
    directory, just the wrong one, so the failure surfaced as "could not find
    examples/" rather than as a path bug. Searching for a marker cannot go
    stale that way.

    Returns None when DKX is installed from a wheel with no checkout above it,
    which is the normal case for a pip user; callers must handle that rather
    than assume a repository is present.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / _CHECKOUT_MARKER).is_file():
            return candidate
    return None
