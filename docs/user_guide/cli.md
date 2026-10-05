# Command line

The `dkx` command has twelve user commands. Everything specific to SFINCS
files lives under `dkx sfincs`, so `dkx --help` stays short.

```text
dkx doctor              check that this install can run
dkx template              print the case template or JSON Schema
dkx validate CASE       validate a case and print its ID
dkx run CASE            solve one case
dkx scan CASE           solve every point of a [scan]
dkx converge CASE       refine each phase-space axis and report convergence
dkx roots RESULT        print the ambipolar root table
dkx inspect RESULT      print what a result contains
dkx plot RESULT         write a figure
dkx compare A B         compare two results
dkx convert DECK CASE   turn a SFINCS namelist into a case file
dkx sfincs ...          SFINCS compatibility commands
```

Two shortcuts need no command name: `dkx input.namelist [options]` runs
`dkx sfincs write-output --input input.namelist [options]`, and
`dkx --plot FILE` (or `dkx FILE.h5`, `dkx FILE.nc`) plots a SFINCS output file,
or, when the file is a VMEC equilibrium, runs a representative survey
([below](#from-an-equilibrium)).

## Options every command accepts

| option | effect |
|---|---|
| `-v`, `--verbose` | more output (repeatable); at `-v` the CLI prints the parallel runtime it applied |
| `-q`, `--quiet` | minimal output; progress is suppressed |
| `--cores N` | CPU threads for the solver, applied before JAX initializes; `0` lets XLA choose; when omitted the thread pool is clamped to `min(8, cpu_count)` |
| `--fortran-stdout` / `--no-fortran-stdout` | mirror or silence the SFINCS v3 progress lines |
| `--transport-workers N` | worker processes for independent transport right-hand sides (namelist route) |
| `--distributed`, `--process-id`, `--process-count`, `--coordinator-address`, `--coordinator-port` | JAX multi-host initialization |

`--cores` exists because XLA sizes its host thread pool once, when the CPU
backend starts; the CLI sets it and restarts itself if JAX was already
imported. The clamp to 8 reflects a measured optimum of 4–8 threads, with a
full-width pool on a many-core host slower than 8 threads
({doc}`scans_and_parallelism`).

## Exit status

| code | meaning |
|---|---|
| `0` | success |
| `1` | the command ran and the answer is "no": a scan point failed, `converge` found the case unresolved or a residual unaccepted, `doctor` found a blocking problem |
| `2` | the command could not run: a missing file, an invalid case, an unsupported model; `compare` also returns `2` when the results differ |

## Starting out

### `dkx doctor`

```console
dkx doctor [--format table|json]
```

Checks Python, DKX, SOLVAX and the required packages, whether JAX arrays
actually materialize in float64, and which devices JAX sees
({doc}`../getting_started/installation`). Exits 1 on a blocking failure.

### `dkx template`

```console
dkx template [--format toml|json]
```

`toml` (default) prints a commented case using every field; `json` prints the
JSON Schema (draft 2020-12) for editors and validators. The old name
`dkx schema` still works as a hidden alias for one release.

### `dkx validate`

```console
dkx validate CASE
```

Parses a `.toml` or `.json` case, runs the executor's preflight (the options
the native route refuses; see {doc}`inputs`), and prints the name, the
`case_id`, the workflow and the counts. For a `[scan]` case it prints the
expanded case count; for an ambipolar case, the bound on retained evaluations
and bytes. It solves nothing and does not open the geometry file. Exits 2 with
the field path, value, expected form and correction on failure.

## Running

### `dkx run`

```console
dkx run CASE [--out RESULT.nc]
```

Solves one case and prints the workflow, convergence flag, true residual,
solver route, wall time and output path, then the result summary. It writes the
NetCDF result to `--out`, or to `[output].file` resolved beside the case file,
plus a `.png` summary beside it when `[output].plots = true`.
Progress goes to stderr. A case with `[scan]` is refused; use `dkx scan`.

### `dkx scan`

```console
dkx scan CASE [--out SCAN.nc] [--no-resume]
```

Expands `[scan]` into derived cases, solves each, and writes one result with a
leading `case` dimension ({doc}`outputs`). A failing point is recorded as
`failed: ...` and the scan continues; the file is still written and the command
exits 1. Resumption skips points whose `case_id` is already in the output, so a
scan resumed after the case was edited reruns the changed points instead of
mixing physics. `--no-resume` reruns everything. `--out` overrides
`scan.output`.

### `dkx converge`

```console
dkx converge CASE [--axes theta zeta pitch speed] [--factor 1.5]
                  [--tolerance 0.02] [--no-joint] [--format table|json]
```

Refines each named phase-space axis by `--factor` (at least one grid point),
then all of them together, and reports the worst relative change of the
observables. `CASE` is a native case or a linear `.namelist` deck. Exits 0 only
when every change is below `--tolerance` **and** every solve's original
residual was accepted. `--no-joint` skips the joint run, which is faster and
cannot detect coupling between axes. {doc}`convergence` explains the report.

## Reading results

| command | options | does |
|---|---|---|
| `dkx inspect RESULT` | | prints workflow, schema, convergence and every array's shape and dtype, without recomputing |
| `dkx plot RESULT` | `--out PATH`, `--kind auto\|summary\|search\|scan` | writes a PNG (default beside the input) |
| `dkx roots RESULT` | `--format table\|json` | prints ambipolar roots with classification, bracket width and branch, and flags surfaces with nonsmooth branch events |
| `dkx compare A B` | `--rtol 1e-9`, `--atol 0`, `--verbose-keys`, `--format table\|json` | compares every array; exits 2 on any difference |

`dkx plot --kind auto` picks a radial-profile panel for a profile or ambipolar
result and an observables-against-axis panel for a scan. `--kind search` draws
the radial current against $E_r$ with every evaluation, bracket, root type and
the selected branch; it is the panel to look at before trusting a root. A
failed evaluation is marked on the axis rather than drawn at $J_r = 0$. For a
SFINCS output file only the summary panel exists.

`dkx roots` reports why a surface without roots is not proof that none exist:
sign sampling cannot see a tangential root or an even number of crossings
between two samples. A nonsmooth branch event means the profile is not
differentiable there, even though `jax.grad` returns a number.

`dkx compare` treats a pair of `.h5` files as SFINCS output (with the upstream
per-dataset tolerances) and anything else as DKX results, and refuses a mixed
pair. Wall time and iteration counts are shown but never decide the verdict.

## Coming from SFINCS

### `dkx convert`

```console
dkx convert DECK DESTINATION [--name NAME] [--force]
```

Writes the native case a SFINCS deck describes; the destination extension
(`.toml` or `.json`) picks the format, and `--force` overwrites. The deck's
single surface becomes three surfaces with a profile linear in $\hat r$. A deck
using a model the native route does not implement is refused at convert time,
naming the key ({doc}`sfincs_namelist`).

### `dkx sfincs`

The compatibility commands read and write SFINCS files directly, without
conversion. `dkx sfincs --help` lists them.

| command | purpose | main options |
|---|---|---|
| `write-output` | solve a deck and write `sfincsOutput` | `--input`, `--out` (`.h5`, `.nc`, `.npz`), `--solve-method`, `--geometry-only`, `--compute-transport-matrix`, `--compute-solution`, `--solver-trace JSON`, `--no-fortran-layout`, `--no-overwrite`, `--equilibrium-file`/`--wout-path` |
| `solve-v3` | solve a deck and write the state vector | `--input`, `--out-state`, `--tol`, `--atol`, `--maxiter`, `--which-rhs`, `--solve-method` |
| `transport-matrix-v3` | `RHSMode = 2/3` transport matrix | `--input`, `--out-matrix` (`.npy`), `--out`, `--out-state-prefix`, `--tol`, `--solve-method` |
| `monoenergetic-database` | $D^*_{ij}(\nu', E^*)$ database | `--input`, `--nu-prime ...`, `--e-star ...`, `--out` (`.npz`), `--tol` |
| `ambipolar` | in-process Brent ambipolar $E_r$ solve | `--input`, `--out-dir`, `--er-min`, `--er-max`, `--er-initial`, `--max-evaluations`, `--current-tolerance` |
| `scan-er` | write one `sfincsOutput.h5` per $E_r$ value | `--input`, `--out-dir`, `--min`/`--max`/`--n` or `--values`, `--jobs`, `--index`/`--stride`, `--compute-transport-matrix` |
| `ambipolar-solve` | find roots in a `scan-er` directory | `--scan-dir`, `--n-fine` |
| `run-fortran` | run a compiled SFINCS v3 executable | `--input`, `--exe`, `--workdir` |
| `compare-h5` | compare two SFINCS HDF5 files | `--a`, `--b`, `--rtol`, `--atol`, `--tolerances-json`, `--show-all` |
| `dump-h5` | dump an HDF5 file to JSON | `--sfincs-output`, `--out-json`, `--keys-only` |
| `plot-output` | diagnostics panel from an output file | `--input-h5`, `--out` |
| `postprocess-upstream` | run an upstream `utils/` script | `--case-dir`, `--util`, `--utils-dir` |

`--solve-method` defaults to `auto`, which is the recommended setting
({doc}`../numerics/solver_routes`). `--equilibrium-file` (and its alias
`--wout-path`) replaces the deck's `equilibriumFile` without editing the deck.

Every `dkx sfincs` command is also accepted at the top level
(`dkx write-output`, `dkx compare-h5`, ...) as a legacy alias hidden from
`dkx --help`.

## From an equilibrium

```console
dkx wout_XXX.nc                        # density scaled from p(0)
dkx wout_XXX.nc --density-m3 2.38e20   # pin the on-axis density; T(0) follows
dkx wout_XXX.nc --quick                # coarser, for a first look
```

Given a VMEC `wout` file, `dkx` runs a representative ambipolar survey and
writes a panel. A VMEC equilibrium fixes only the pressure, so the survey has
to assume a plasma: it scales the on-axis density from the pressure against
the reactor profiles of Landreman, Buller and Drevlak (arXiv:2205.02914),
takes the temperature from $p = 2nT$, and prints the pair it used. The
bootstrap current depends strongly on that choice, because collisionality
goes like $n/T^2$ at fixed pressure; pin `--density-m3` to the design point
before comparing with an optimizer. The survey is a first look, not a study:
for results, write a case with the real profiles, then `dkx validate` and
`dkx converge` it.

Each surface of the radial scan is solved with that surface's own $n$, $T$
and gradients. The evaluated field is the most negative *stable* root of
$J_r(E_r)$; stability comes from the outward-current slope, as in
`dkx.er`. The ambipolarity panel labels every root ion, unstable or
electron.
The evaluated root is refined from its scan bracket by Brent's method (at
most four more solves per surface) and the moments are read there, not
interpolated across the bracket.

From Python, `dkx.representative.run_representative` takes the plasma
directly instead of assuming one. This is how VMEX's `vmex --neoclassical`
calls it:

```python
from dkx.representative import run_representative

run_representative(
    "wout_XXX.nc",
    surfaces=(0.25, 0.5, 0.75),          # r/a of the radial scan
    er=None,                             # None: ambipolar root; float: prescribed kV/m
    collision_operator="fp",             # "fp" | "pas" | "pas+momentum_correction"
    profiles=kinetic_profiles,           # ne_coeffs [m^-3], Te_coeffs/Ti_coeffs [eV]
    redl_jdotb=(s, jdotb_redl),          # optional external Redl <j.B> [A T/m^2]
)
```

`profiles` is any object or mapping with `ne_coeffs`, `Te_coeffs` and
`Ti_coeffs`, polynomials in $s$ with the lowest order first (VMEX's
`KineticProfiles`). It replaces the pressure split, so the kinetic and
equilibrium currents then describe the same plasma, and $T_i \ne T_e$ is
allowed. DKX has no Redl implementation of its own. `redl_jdotb` draws a
caller's Redl curve on the bootstrap panel. With
`"pas+momentum_correction"`, the $E_r$ scan uses pitch-angle scattering, and
the bootstrap current at the evaluated field is corrected by the
Sugama–Nishimura moment method. The defaults reproduce `dkx wout_XXX.nc`.
