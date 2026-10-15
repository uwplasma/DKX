"""All of the ``dkx.solve*`` modules as one namespace, for tests that patch internals.

``dkx.solve`` is split by route; a helper a test patches may be looked up from
more than one of them. Reading an attribute returns the first module's copy;
setting one rebinds it in every module that holds it, so a patch reaches every
caller and ``monkeypatch`` undoes it everywhere.
"""

import importlib

_MODULES = [
    importlib.import_module(f"dkx.{name}")
    for name in ("solve", "solve_krylov", "solve_direct", "solve_structured", "solve_common")
]


class _Solve:
    def __getattr__(self, name):
        for module in _MODULES:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        held = [module for module in _MODULES if hasattr(module, name)]
        if not held:
            raise AttributeError(name)
        for module in held:
            setattr(module, name, value)


solve_modules = _Solve()
