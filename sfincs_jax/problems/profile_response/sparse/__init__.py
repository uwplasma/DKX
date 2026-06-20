"""Sparse profile-response solver stages."""

from . import direct as direct
from . import finalization as finalization
from . import fortran_reduced as fortran_reduced
from . import krylov as krylov
from . import qi as qi
from . import xblock as xblock

__all__ = (
    *direct.__all__,
    *finalization.__all__,
    *fortran_reduced.__all__,
    *krylov.__all__,
    *qi.__all__,
    *xblock.__all__,
)

for _module in (direct, finalization, fortran_reduced, krylov, qi, xblock):
    for _name in _module.__all__:
        globals()[_name] = getattr(_module, _name)

del _module
del _name
