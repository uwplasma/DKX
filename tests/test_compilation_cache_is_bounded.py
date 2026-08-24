"""The persistent compilation cache must be bounded.

dkx sets the persistent-cache size and compile-time thresholds to zero so that
even fast-compiling kernels are cached.  That makes every distinct grid a scan
touches leave an entry behind, and nothing removes one: uncapped, the cache
reached 782 MB over 64k files on a development machine.  Users never see it
grow, so the bound has to be the default rather than advice in a doc.
"""

import os
import subprocess
import sys

FOUR_GIB = 4 * 1024**3


def _cache_max_size_in_a_fresh_process(env_extra: dict[str, str]) -> int:
    """Import dkx in a subprocess and report the cap it installed."""
    env = {**os.environ, **env_extra}
    code = (
        "import dkx, jax; "
        "print(int(jax.config.jax_compilation_cache_max_size))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    return int(out.stdout.strip().splitlines()[-1])


def test_importing_dkx_bounds_the_cache_by_default() -> None:
    assert _cache_max_size_in_a_fresh_process({}) == FOUR_GIB


def test_the_bound_is_overridable() -> None:
    assert _cache_max_size_in_a_fresh_process(
        {"DKX_COMPILATION_CACHE_MAX_BYTES": str(512 * 1024**2)}
    ) == 512 * 1024**2


def test_zero_disables_the_bound() -> None:
    """0 means 'do not cap', for a user who manages the cache themselves."""
    assert _cache_max_size_in_a_fresh_process(
        {"DKX_COMPILATION_CACHE_MAX_BYTES": "0"}
    ) != FOUR_GIB
