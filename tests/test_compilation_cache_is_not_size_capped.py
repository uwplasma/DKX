"""The persistent compilation cache is deliberately left uncapped.

Capping it looks obviously right -- it had reached 782 MB over 64k entries on
a development machine -- and it is wrong.  Setting
``jax_compilation_cache_max_size`` turns on jax's LRU, which writes a sidecar
``-atime`` file per entry and does size bookkeeping on every write.  Combined
with the zero min-entry-size/min-compile-time thresholds dkx sets on purpose,
so that even small kernels are cached, that costs about 60%: measured on
tests/test_monoenergetic_database.py, 32 s and 1792 files capped against 20 s
and 896 uncapped.  CI showed the same thing at scale -- nine of ten coverage
shards crossed a 10-minute timeout they had been finishing in four to eight.

An unbounded cache costs disk, and deleting the directory costs only compile
time.  Paying 60% on every run to avoid that is the worse trade, so this test
exists to stop the cap being reintroduced as an obvious improvement.
"""

import os
import subprocess
import sys


def _in_a_fresh_process(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=os.environ, check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_configuring_the_runtime_does_not_cap_the_cache() -> None:
    capped = _in_a_fresh_process(
        "import dkx.run, jax; print(jax.config.jax_compilation_cache_max_size)"
    )
    assert int(capped) <= 0, (
        "dkx must not set jax_compilation_cache_max_size: it enables jax's LRU, "
        "which costs ~60% runtime with dkx's zero cache thresholds"
    )


def test_the_cache_is_still_enabled() -> None:
    """Not capping is not the same as not caching; the cache must still work.

    The trigger is reaching for the solve stack, not importing the package:
    ``import dkx`` is inert now, and the cache directory is one of the seven
    things it stopped doing (plan.md 6.4).
    """
    directory = _in_a_fresh_process(
        "import dkx.run, os; print(os.environ.get('JAX_COMPILATION_CACHE_DIR', ''))"
    )
    assert directory.endswith("jax_compilation_cache"), directory


def test_solving_does_not_warn_about_the_cache(tmp_path) -> None:
    """A run must not emit cache-write warnings, whatever the cache settings."""
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
