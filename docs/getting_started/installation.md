# Installation

DKX is a Python package built on JAX. It needs Python 3.11 or later.

```console
pip install dkx
```

The install pulls in everything a run needs: `jax`, `numpy>=2.1`, `scipy`,
`h5py`, `netCDF4`, `matplotlib`, `rich` and `solvax>=0.28.0`
(`pyproject.toml`). No optional extra is required to solve a case, write a
NetCDF or HDF5 file, or plot one.

## Check the install

```console
dkx doctor
```

`dkx doctor` prints one row per check and exits non-zero when any blocking
check fails. It reports what the process observes rather than what the
environment requests:

| check | fails when |
|---|---|
| `python` | the interpreter version is below 3.11 |
| `dkx` | warns (does not fail) when running from a source checkout rather than an installed distribution |
| `solvax` | missing, or below the 0.24.0 floor |
| `jax`, `jaxlib`, `numpy`, `scipy`, `rich` | missing |
| `h5py`, `netCDF4`, `matplotlib` | missing (warning only) |
| `float64` | a JAX array requested as `float64` materializes as another dtype |
| `devices` | JAX enumerates no devices |

`dkx doctor --format json` prints the same rows as JSON.

## The solver library

The structured linear-algebra routes (block-tridiagonal Legendre elimination,
recycled Krylov, implicit differentiation, sparse assembly by compression) live
in the external [solvax](https://pypi.org/project/solvax/) package. It is a
core dependency and installs with DKX. The floor is 0.24.0 because that release
adds the operator equilibration applied before a factorization
(`pyproject.toml`). The sparse direct route can use MUMPS instead of SuperLU
only with SOLVAX 0.25.0 or later and PyMUMPS installed; see
{doc}`python_api <../user_guide/python_api>`.

## GPU

Install the CUDA build of JAX that matches the driver, for example

```console
pip install -U "jax[cuda12]"
```

No DKX change is needed; the same solves run on the accelerator. A single
solve gains little on a GPU; batched independent solves are where it helps
({doc}`../user_guide/scans_and_parallelism`).

## Floating-point precision

DKX requires JAX in float64. Importing `dkx` enables it once for the process.
A program that shares the interpreter with a library tuned for float32 can
decline that global change:

```console
DKX_NO_X64_SETUP=1 python my_script.py
```

Declining the setting does not remove the requirement: `dkx.require_float64()`
runs on the solve path and raises, naming the ways to fix it, so a process with
float64 off cannot silently return single-precision answers. Setting
`JAX_ENABLE_X64=1` explicitly works and takes precedence over the opt-out.

## Equilibrium data

Multi-megabyte public VMEC and Boozer files are not stored in the repository or
the wheel. Examples and tests that need the W7-X, HSX or QI equilibria
resolve them by basename and fetch the `dkx-data-v1` GitHub release asset into
a user cache on first use. The default cache root is `~/.cache/dkx/data`.

```console
python -m dkx.validation.data_fetch     # prefetch every release file
```

| variable | effect |
|---|---|
| `DKX_DATA_DIR` | cache root for release-hosted equilibrium files |
| `DKX_OFFLINE=1` | an uncached file raises instead of downloading (CI, cluster jobs) |
| `DKX_EQUILIBRIA_DIRS` | extra directories searched for equilibrium files named by a SFINCS deck |

## Development and documentation installs

```console
git clone https://github.com/uwplasma/DKX
cd DKX
pip install -e ".[dev]"      # pytest, pytest-xdist, pytest-split, pypdf
pip install -e ".[docs]"     # sphinx, myst-parser, furo
```

Some optimization examples use `optax`; install it separately when running
them.

## SFINCS Fortran reference build (optional)

Parity tooling can compare DKX against a local SFINCS Fortran v3 executable.
DKX itself never needs it: frozen reference outputs ship with the test
fixtures. A reproducible build on macOS or Linux uses a conda environment that
provides PETSc and MUMPS together with upstream's `makefiles/makefile.conda`
in `fortran/version3` of the SFINCS repository. `dkx sfincs run-fortran`
launches such an executable on a namelist ({doc}`../user_guide/cli`).
