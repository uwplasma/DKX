# Parity

One script that compares a DKX `sfincsOutput.h5` against the frozen SFINCS
Fortran v3 fixture in `tests/ref/`, and one Fortran control inventory.

- `output_parity_cli_driver.py`: runs the case through the `dkx` CLI and compares
  it dataset by dataset;
  exits non-zero on a mismatch. CI runs it.
- `output_key_coverage_report.py`: output-key coverage and the Fortran
  namelist-control inventory (`--namelist-source`), imported by `tests/test_validation.py`.

The operator, residual, and solve parity against the frozen PETSc binaries is
asserted by `tests/test_kinetic_operator_fortran_parity.py`; output parity by
`tests/test_output_h5_scheme4_parity.py`.
