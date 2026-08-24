"""dkx must name the real problem when numpy is too old for jax.

A user installed dkx into the Anaconda base environment, where conda already
owned numpy 1.x.  pip resolved happily, then the first ``import jax`` died with
``NameError: name 'isnan' is not defined`` from inside
``numpy/core/getlimits.py``, raised while ml_dtypes probed bfloat16's finfo.
That traceback names neither dkx nor numpy's version.  CI never reproduces it
because CI always installs into a clean environment.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _import_dkx_with_numpy_version(version: str) -> subprocess.CompletedProcess:
    """Import dkx in a fresh interpreter with numpy reporting ``version``."""
    code = textwrap.dedent(
        f"""
        import sys, types
        fake = types.ModuleType("numpy")
        fake.__version__ = {version!r}
        sys.modules["numpy"] = fake
        try:
            import dkx
        except ImportError as exc:
            print("IMPORTERROR", exc)
        else:
            print("IMPORTED")
        """
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))  # fmt: skip


@pytest.mark.parametrize("version", ["1.21.6", "1.26.4"])
def test_numpy_1x_is_refused_with_the_fix_in_the_message(version: str) -> None:
    out = _import_dkx_with_numpy_version(version).stdout
    assert "IMPORTERROR" in out, f"numpy {version} was accepted; it cannot work"
    assert "numpy>=2.1" in out, "the message must state the requirement"
    assert version in out, "the message must say what was actually found"
    # The whole point is that the user can act on it without a web search.
    assert "pip install -U" in out and "conda create" in out


def test_the_guard_runs_before_jax_is_imported() -> None:
    """A numpy check that only fires after `import jax` is no check at all."""
    source = (__import__("pathlib").Path(__file__).resolve().parents[1]
              / "dkx" / "__init__.py").read_text(encoding="utf-8")  # fmt: skip
    assert source.index("_check_numpy()") < source.index("import jax"), (
        "the numpy guard must run before any jax import, or the cryptic "
        "ml_dtypes traceback wins the race"
    )
