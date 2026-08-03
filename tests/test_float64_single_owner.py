"""One module decides JAX precision, and a caller who overrides it is told.

float64 is a correctness requirement for `dkx`: the block eliminations and
every parity fixture depend on it, and single precision changes which results
are trustworthy rather than merely how accurate they are.

The requirement is not in question; the mechanism was.  Sixteen modules used to
call ``jax.config.update("jax_enable_x64", True)`` at module scope, so importing
any part of `dkx` made a global, invisible, import-order-dependent change to
the precision of every other JAX library in the process.  See uwplasma/DKX#22.

These tests pin the arrangement that replaced it: one owner, an opt-out for
callers who manage precision themselves, and a loud check on the solve path so
opting out cannot silently produce float32 answers.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run(code: str, **env_extra: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter; precision is process-global.

    ``JAX_ENABLE_X64`` is dropped from the child environment.  ``conftest``
    exports it for the test session, and a child that inherits it gets float64
    whatever ``dkx`` decides -- correctly, since an explicit request from the
    user outranks the package default, but it would make the opt-out tests
    below assert nothing.
    """
    import os

    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    env.update({"PYTHONPATH": str(REPO_ROOT), **env_extra})
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env,
        timeout=300, check=False,
    )


def test_only_one_module_sets_the_global_precision():
    """The count is the point: sixteen incidental owners became one.

    A new module quietly adding the call back would restore exactly the
    import-order surprise this arrangement removed, so the check is on the set
    of files rather than on behaviour.
    """
    setters = {
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "dkx").rglob("*.py")
        if 'update("jax_enable_x64"' in path.read_text()
    }
    assert setters == {pathlib.Path("dkx/__init__.py")}, setters


def test_importing_dkx_gives_float64_by_default():
    """The convenience the previous arrangement provided is kept."""
    result = _run("import jax.numpy as jnp, dkx; print(jnp.zeros(1).dtype)")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "float64" in result.stdout


def test_a_caller_can_opt_out():
    """``DKX_NO_X64_SETUP`` hands precision back to the caller.

    This is the case the issue was filed for: a process that imports `dkx`
    alongside a library tuned for float32 should be able to say no.
    """
    result = _run(
        "import jax.numpy as jnp, dkx; print(jnp.zeros(1).dtype)",
        DKX_NO_X64_SETUP="1",
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "float32" in result.stdout


def test_opting_out_does_not_opt_into_wrong_answers():
    """``require_float64`` raises, naming every way to fix it.

    Opting out of the *setting* must not opt out of the *requirement*; without
    this the escape hatch would trade a loud global side effect for a silent
    single-precision solve, which is the worse of the two.
    """
    result = _run(
        "import dkx\n"
        "try:\n"
        "    dkx.require_float64()\n"
        "except RuntimeError as exc:\n"
        "    print('RAISED', exc)\n",
        DKX_NO_X64_SETUP="1",
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "RAISED" in result.stdout
    for hint in ("jax_enable_x64", "JAX_ENABLE_X64", "DKX_NO_X64_SETUP"):
        assert hint in result.stdout, hint


def test_the_check_probes_the_dtype_not_the_config_flag():
    """What matters is the dtype arrays actually get.

    The config can be set and later overridden; a flag read would pass while
    the arrays were still single precision.
    """
    source = (REPO_ROOT / "dkx" / "__init__.py").read_text()
    body = source[source.index("def require_float64"):]
    body = body[: body.index("\ndef ") if "\ndef " in body[1:] else len(body)]
    assert "zeros(1).dtype" in body


def test_the_solve_entry_point_enforces_it():
    """The funnel every solve passes through, so no route escapes the check."""
    solve_source = (REPO_ROOT / "dkx" / "solve.py").read_text()
    assert "require_float64()" in solve_source


@pytest.mark.parametrize("module", ["dkx.solve", "dkx.run", "dkx.moments"])
def test_submodules_still_import_standalone(module: str):
    """Removing the module-scope call must not break direct submodule imports."""
    result = _run(f"import {module}; print('ok')")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout


def test_an_explicit_user_request_outranks_the_opt_out():
    """``JAX_ENABLE_X64=1`` with ``DKX_NO_X64_SETUP=1`` still gives float64.

    The opt-out says "dkx must not decide", not "float64 must be off".  A user
    who asked JAX directly gets what they asked for.
    """
    import os

    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    env.update({
        "PYTHONPATH": str(REPO_ROOT),
        "DKX_NO_X64_SETUP": "1",
        "JAX_ENABLE_X64": "1",
    })
    result = subprocess.run(
        [sys.executable, "-c", "import jax.numpy as jnp, dkx; print(jnp.zeros(1).dtype)"],
        capture_output=True, text=True, env=env, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "float64" in result.stdout
