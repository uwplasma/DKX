# Coming from SFINCS

DKX reads SFINCS v3 `input.namelist` decks directly and writes
`sfincsOutput.h5` with SFINCS's variable names, so an existing deck, scan
directory or post-processing script can be used unchanged. It also has a native
case format, in SI units and with whole profiles instead of one surface; this
tutorial shows how to run a deck as it is, how to convert it, and how to compare
the two codes' outputs. The namelist reference is
{doc}`../user_guide/sfincs_namelist`.

## Run a deck unchanged

```console
dkx input.namelist
```

With no subcommand, `dkx` treats its argument as a namelist and writes
`sfincsOutput.h5` in the working directory, the way the Fortran executable does. The explicit
form, with the output format chosen by the file suffix (`.h5`, `.nc` or `.npz`):

```console
dkx sfincs write-output --input input.namelist --out sfincsOutput.h5
```

The SFINCS-compatibility commands are grouped under `dkx sfincs` (and are also
accepted at the top level):

| Command | Does |
| --- | --- |
| `write-output` | run a deck and write `sfincsOutput.h5` (or NetCDF, NPZ) |
| `transport-matrix-v3` | solve an `RHSMode = 2/3` deck and write `transportMatrix.npy` |
| `monoenergetic-database` | scan $(\nu', E^*)$ and write a monoenergetic database |
| `scan-er` | an $E_r$ scan into one run directory per point, like `sfincsScan` |
| `ambipolar-solve` | find the ambipolar roots of a finished `scan-er` directory |
| `ambipolar` | an in-process Brent ambipolar solve from a deck |
| `solve-v3` | solve and write the raw `stateVector.npy` |
| `run-fortran` | run a compiled SFINCS v3 executable, for comparison |
| `dump-h5`, `compare-h5`, `plot-output` | inspect, compare and plot SFINCS HDF5 files |
| `postprocess-upstream` | run one of the upstream `utils/` scripts on the output |

From Python, `dkx.load_sfincs_input("input.namelist")` returns a validated,
typed `SfincsInput`; `dkx.run_from_namelist(path, out_path="sfincsOutput.h5")`
runs it.

Every deck of the upstream `fortran/version3/examples` suite is vendored in
`examples/sfincs_examples/`. Run them all, or a subset, and optionally the
Fortran executable beside them:

```console
python examples/sfincs_examples/run_dkx.py --write-output
python examples/sfincs_examples/run_dkx.py --write-output --pattern monoenergetic_geometryScheme11
python examples/sfincs_examples/run_dkx.py --write-output --compare-fortran --fortran-exe ../sfincs/fortran/version3/sfincs
```

## What to expect

On the upstream suite the two codes agree to a median relative difference of
4.1e-6 on shared outputs, and full Fokker–Planck decks to 1e-8
({doc}`../benchmarks/sfincs`). Three differences are worth knowing before
comparing numbers:

- **Solver.** DKX does not use PETSc or MUMPS. It picks a structured direct,
  sparse direct or recycled Krylov route from the operator's structure and
  reports the residual of the original equation for every solve
  ({doc}`../numerics/solver_routes`). Solver options in the deck that select
  PETSc behaviour have no effect on the answer.
- **Sparsification.** Released SFINCS drops matrix entries with magnitude at or
  below 1e-12. On cold-ion, hot-electron decks this removes part of the
  ion–electron collision coupling and moves the bootstrap current by 12–19% on
  the HSX case documented in {doc}`../benchmarks/sfincs`. DKX keeps every entry.
  Compare against a SFINCS build with the threshold set to zero on such decks.
- **Scans.** `sfincsScan` launches one process per point, and so does
  `dkx sfincs scan-er`, which pays compilation on every point. Inside one Python
  process, `dkx.batched_er_scan` and the ambipolar solver reuse compiled code,
  preconditioners and the Krylov recycle subspace across points
  ({doc}`../user_guide/scans_and_parallelism`).

## Convert to a native case

```console
dkx convert input.namelist case.toml
dkx validate case.toml
dkx run case.toml --out result.nc
```

The extension of the destination picks TOML or JSON; `--name` sets the case
name and `--force` overwrites. A deck is dimensionless and describes one
surface; a case is in SI and profile-shaped, so the conversion is a translation
of units and conventions rather than a renaming. Anything the case schema
cannot carry makes the conversion fail with the namelist key named, rather than
producing a case that runs and answers a different question. The main
limitations of the native executor are:

| Deck feature | Native case | Use instead |
| --- | --- | --- |
| `RHSMode = 2` or `3` | refused | `dkx sfincs transport-matrix-v3`, `dkx sfincs monoenergetic-database` ({doc}`transport_matrix`) |
| $\Phi_1$ in the kinetic equation or quasineutrality | refused; the executor implements `phi1 = "off"` | the namelist route, as `examples/09_phi1_and_impurities` does |
| RHSMode 4/5 adjoint sensitivities | not in the case schema | the namelist route and `dkx.sensitivity` |

Multi-species runs without $\Phi_1$ are fully native. The native result is a
NetCDF file with SI arrays such as `particle_flux_m2_s` and `heat_flux_W_m2`
({doc}`../user_guide/outputs`); `dkx inspect` lists them.

## Compare outputs

```console
dkx compare sfincsOutput_fortran.h5 sfincsOutput.h5 --rtol 1e-9
```

`dkx compare` reads DKX NetCDF results and SFINCS HDF5 files, prints the arrays
that differ (`--verbose-keys` for all of them, `--format json` for a machine
report), and exits non-zero if any exceed the tolerance, so it can gate a
script or CI job. `dkx sfincs compare-h5` is the HDF5-only form.

Plotting scripts written for SFINCS output keep working:

```console
dkx sfincs postprocess-upstream --case-dir path/to/case --util sfincsScanPlot_1 -- pdf
```

The upstream scripts read defaults from `globalVariables.F90`, which is vendored
as `examples/sfincs_examples/globalVariables.F90`; outside a repository
checkout, set `DKX_UPSTREAM_UTILS_DIR` to the `utils` directory.

```{figure} ../_static/figures/utils/sfincsScanPlot_1.png
:alt: Upstream sfincsScanPlot_1 output generated from DKX sfincsOutput.h5 files.
:width: 75%

An upstream `sfincsScanPlot_1` figure drawn from DKX output.
```
