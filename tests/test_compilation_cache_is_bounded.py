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


def test_the_cache_directory_is_versioned_away_from_the_uncapped_one() -> None:
    """A pre-cap cache directory cannot be LRU-managed, so do not try.

    JAX's LRU keeps a sidecar "-atime" file per entry.  A directory filled in
    before the cap existed has none, so every eviction pass tries to touch a
    file that was never written: 12 warnings on one small solve against a
    legacy cache, 0 against a fresh one.  Bounding the cache therefore means
    starting a new directory rather than adopting the old one.
    """
    out = subprocess.run(
        [sys.executable, "-c", "import dkx, os; print(os.environ['JAX_COMPILATION_CACHE_DIR'])"],
        capture_output=True, text=True, env=os.environ, check=True,
    )
    directory = out.stdout.strip().splitlines()[-1]
    assert directory.endswith("jax_compilation_cache_v2"), directory


def test_solving_does_not_warn_about_cache_writes(tmp_path) -> None:
    """The bound must not buy itself a stream of warnings on every run."""
    env = {
        **os.environ,
        "DKX_COMPILATION_CACHE_DIR": str(tmp_path / "cache"),
        "PYTHONWARNINGS": "always",
    }
    code = (
        "import dkx\n"
        "dkx.run(geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,\n"
        "        B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0, iota=0.4542,\n"
        "        GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,\n"
        "        Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],\n"
        "        dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],\n"
        "        Ntheta=9, Nzeta=1, Nxi=8, NL=4, Nx=4,\n"
        "        collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=0.01)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "persistent compilation cache" not in out.stderr, out.stderr[-2000:]
