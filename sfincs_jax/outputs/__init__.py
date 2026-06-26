"""Output schema and file-format helpers for SFINCS_JAX."""

from __future__ import annotations

from .formats import (
    decode_if_bytes,
    fortran_h5_layout,
    output_file_format,
    read_sfincs_h5,
    read_sfincs_output_file,
    to_numpy_for_h5,
    write_sfincs_h5,
    write_sfincs_netcdf,
    write_sfincs_npz,
    write_sfincs_output_file,
)
from .transport import (
    TransportStreamingOutputAccumulator,
    conversion_factors_to_from_dpsi_hat,
    transport_solver_diagnostic_arrays,
    write_transport_h5_streaming,
)
from .writer import (
    ExportFConfig,
    localize_equilibrium_file_in_place,
    sfincs_jax_output_dict,
    write_sfincs_jax_output_h5,
)

__all__ = (
    "ExportFConfig",
    "TransportStreamingOutputAccumulator",
    "conversion_factors_to_from_dpsi_hat",
    "decode_if_bytes",
    "fortran_h5_layout",
    "localize_equilibrium_file_in_place",
    "output_file_format",
    "read_sfincs_h5",
    "read_sfincs_output_file",
    "sfincs_jax_output_dict",
    "to_numpy_for_h5",
    "transport_solver_diagnostic_arrays",
    "write_transport_h5_streaming",
    "write_sfincs_h5",
    "write_sfincs_jax_output_h5",
    "write_sfincs_netcdf",
    "write_sfincs_npz",
    "write_sfincs_output_file",
)
