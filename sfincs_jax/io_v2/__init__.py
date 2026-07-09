"""SFINCS v3 input/output compatibility layer (v2 refactor target: ``io/``).

- :mod:`sfincs_jax.io_v2.namelist` — ``input.namelist`` parsing, Fortran
  defaults (``globalVariables.F90``) and validation (``validateInput.F90``).
- :mod:`sfincs_jax.io_v2.prints` — Fortran-parity stdout blocks, golden-tested
  against ``reference-data-v2/*/stdout.log``.
"""

from .namelist import (
    RawNamelist,
    SfincsInput,
    load_sfincs_input,
    parse_sfincs_input_text,
    read_sfincs_input,
    sfincs_input_from_raw,
)
from . import prints

__all__ = [
    "RawNamelist",
    "SfincsInput",
    "load_sfincs_input",
    "parse_sfincs_input_text",
    "read_sfincs_input",
    "sfincs_input_from_raw",
    "prints",
]
