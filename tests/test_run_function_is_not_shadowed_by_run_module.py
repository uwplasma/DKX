"""``dkx.run`` must be callable AND keep its module attributes, in any order.

The package exports ``run`` and also contains a ``run`` module.  Python binds
a submodule onto its package when anything imports it, so::

    import dkx
    from dkx.run import profile_moments_from_operator
    dkx.run(case)          # 'module' object is not callable

failed while the same two lines in the other order worked.  Order-dependent
and silent.

Resolving it in favour of the *function* is equally wrong and CI proved it:
seventeen tests do ``monkeypatch.setattr("dkx.run.run_profile", ...)``, and a
function has no such attribute.  ``dkx/run.py`` makes itself callable instead,
so ``dkx.run`` is always the module and always invocable.  Each case runs in a
fresh process because this is about import order.
"""

import subprocess
import sys


def _probe(body: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", body], capture_output=True, text=True, check=True
    )
    return out.stdout.strip().splitlines()[-1]


def test_callable_and_attributed_when_the_submodule_is_imported_first() -> None:
    assert _probe(
        "import dkx\n"
        "from dkx.run import profile_moments_from_operator\n"
        "print(callable(dkx.run) and hasattr(dkx.run, 'run_profile'))\n"
    ) == "True"


def test_callable_and_attributed_when_the_attribute_is_touched_first() -> None:
    assert _probe(
        "import dkx\n_ = dkx.run\nimport dkx.run\n"
        "print(callable(dkx.run) and hasattr(dkx.run, 'run_profile'))\n"
    ) == "True"


def test_monkeypatch_by_module_path_still_resolves(monkeypatch) -> None:
    """What CI caught: seventeen tests patch dkx.run.* by dotted path.

    pytest walks dkx -> run -> run_profile by attribute, so resolving
    ``dkx.run`` to the bare function made every one of them raise
    ``'function' object at dkx.run has no attribute 'run_profile'``.
    """
    import dkx.run

    monkeypatch.setattr("dkx.run.run_profile", lambda *a, **k: "patched")
    assert dkx.run.run_profile() == "patched"


def test_the_submodule_is_still_importable_in_its_own_right() -> None:
    assert _probe(
        "from dkx.run import profile_moments_from_operator as f\nprint(callable(f))\n"
    ) == "True"


def test_importing_dkx_does_not_pull_in_the_solve_stack() -> None:
    """The lazy exports exist to keep `import dkx` cheap; keep it that way."""
    assert _probe("import sys, dkx\nprint('dkx.solve' in sys.modules)\n") == "False"
