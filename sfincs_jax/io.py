"""Compatibility facade for SFINCS output reading and writing.

Concrete output schemas, file-format dispatch, and writer implementations live
under :mod:`sfincs_jax.outputs`. Keep this module small so existing user imports
such as ``from sfincs_jax.io import write_sfincs_jax_output_h5`` continue to
work while the public API moves to domain-owned output modules.
"""

from __future__ import annotations

from .outputs.formats import (
    read_sfincs_h5,
    read_sfincs_output_file,
    write_sfincs_h5,
    write_sfincs_netcdf,
    write_sfincs_npz,
    write_sfincs_output_file,
)
from .outputs import writer as _writer
from .outputs.writer import (
    ExportFConfig,
    _resolve_equilibrium_file_from_namelist,
    localize_equilibrium_file_in_place,
    sfincs_jax_output_dict,
    write_sfincs_jax_output_h5,
)


def __getattr__(name: str):
    """Delegate legacy private ``sfincs_jax.io`` names to the output writer."""

    try:
        return getattr(_writer, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

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
