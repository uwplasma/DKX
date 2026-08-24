"""``dkx.run`` must stay the function even after ``dkx.run`` is imported.

The package exports a ``run`` function and also contains a ``run`` module.
Python binds a submodule onto its package when anything imports it, so::

    import dkx
    from dkx.run import profile_moments_from_operator
    dkx.run(case)          # 'module' object is not callable

failed while the same two lines in the other order worked.  Order-dependent
and silent, which reads as the package being broken rather than as an import
ordering problem.  Each case runs in a fresh process because this is about
import order.
"""

import subprocess
import sys


def _probe(body: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", body], capture_output=True, text=True, check=True
    )
    return out.stdout.strip().splitlines()[-1]


def test_run_stays_callable_when_the_submodule_is_imported_first() -> None:
    assert _probe(
        "import dkx\n"
        "from dkx.run import profile_moments_from_operator\n"
        "print(callable(dkx.run))\n"
    ) == "True"


def test_run_stays_callable_when_the_function_is_touched_first() -> None:
    assert _probe(
        "import dkx\n_ = dkx.run\nimport dkx.run\nprint(callable(dkx.run))\n"
    ) == "True"


def test_the_submodule_is_still_importable_in_its_own_right() -> None:
    """Fixing the shadowing must not make `from dkx.run import ...` fail."""
    assert _probe(
        "from dkx.run import profile_moments_from_operator as f\n"
        "print(callable(f))\n"
    ) == "True"


def test_importing_dkx_does_not_pull_in_the_solve_stack() -> None:
    """The lazy exports exist to keep `import dkx` cheap; keep it that way."""
    assert _probe(
        "import sys, dkx\nprint('dkx.solve' in sys.modules)\n"
    ) == "False"
