"""Sparse profile-response solver stages."""

from . import xblock as xblock

__all__ = xblock.__all__

for _name in __all__:
    globals()[_name] = getattr(xblock, _name)

del _name
