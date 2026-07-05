"""Compatibility facade for SFINCS output reading and writing.

Concrete output schemas, file-format dispatch, and writer implementations live
under :mod:`sfincs_jax.outputs`. Keep this module small so existing user imports
such as ``from sfincs_jax.io import write_sfincs_jax_output_h5`` continue to
work while the public API moves to domain-owned output modules.
"""

from __future__ import annotations

import sys
from types import ModuleType

from .outputs.formats import (
    ExportFConfig,
    read_sfincs_h5,
    read_sfincs_output_file,
    write_sfincs_h5,
    write_sfincs_netcdf,
    write_sfincs_npz,
    write_sfincs_output_file,
)
from .outputs import writer as _writer
from .outputs import rhsmode1 as _rhsmode1
from .outputs import formats as _formats
from . import input_compat as _input_compat
from .input_compat import (
    _resolve_equilibrium_file_from_namelist,
    localize_equilibrium_file_in_place,
)
from .outputs.writer import (
    sfincs_jax_output_dict,
    write_sfincs_jax_output_h5,
)

_LEGACY_OWNER_MODULES = (_writer, _rhsmode1, _formats, _input_compat)


def __getattr__(name: str):
    """Delegate legacy private ``sfincs_jax.io`` names to output owners."""

    for module in _LEGACY_OWNER_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _IOFacadeModule(ModuleType):
    """Forward legacy private monkeypatches to output owner modules."""

    def __setattr__(self, name: str, value: object) -> None:
        for module in _LEGACY_OWNER_MODULES:
            if hasattr(module, name):
                setattr(module, name, value)
                break
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _IOFacadeModule

__all__ = (
    "ExportFConfig",
    "_resolve_equilibrium_file_from_namelist",
    "localize_equilibrium_file_in_place",
    "read_sfincs_h5",
    "read_sfincs_output_file",
    "sfincs_jax_output_dict",
    "write_sfincs_h5",
    "write_sfincs_jax_output_h5",
    "write_sfincs_netcdf",
    "write_sfincs_npz",
    "write_sfincs_output_file",
)
