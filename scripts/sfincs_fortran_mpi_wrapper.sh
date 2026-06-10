#!/usr/bin/env bash
set -euo pipefail

# Wrapper for benchmark runners that accept a single executable path.
# Set SFINCS_FORTRAN_EXE to the compiled SFINCS v3 binary and optionally set
# SFINCS_FORTRAN_MPI_NP to the number of MPI ranks. The default is one rank so
# local parity/reference runs avoid concurrent HDF5 output-file writes unless
# MPI scaling is explicitly requested. If mpirun is unavailable, fall back to
# launching the executable directly.

if [[ -z "${SFINCS_FORTRAN_EXE:-}" ]]; then
  echo "SFINCS_FORTRAN_EXE must point to the SFINCS Fortran v3 executable." >&2
  exit 2
fi

if [[ ! -x "${SFINCS_FORTRAN_EXE}" ]]; then
  echo "SFINCS_FORTRAN_EXE is not executable: ${SFINCS_FORTRAN_EXE}" >&2
  exit 2
fi

np="${SFINCS_FORTRAN_MPI_NP:-${SFINCS_MPI_NP:-1}}"

if command -v mpirun >/dev/null 2>&1; then
  exec mpirun -np "${np}" "${SFINCS_FORTRAN_EXE}" "$@"
fi

exec "${SFINCS_FORTRAN_EXE}" "$@"
