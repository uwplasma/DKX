# Python API

Everything the command line does is available from `import dkx`. The names
below are the public surface listed in `dkx.__all__`; {doc}`../api` has the
generated reference.

## Native cases

```python
import dkx

case = dkx.Case.from_file("case.toml")          # or .json
case = dkx.Case.from_mapping({...})             # a dict with the file's structure
result = dkx.run(case)                           # dkx.Result, in memory
result = dkx.run(case, out="result.nc")          # and written to NetCDF
again = dkx.Result.load("result.nc")
```

| name | purpose |
|---|---|
| `Case` | immutable case; `from_file`, `from_mapping`, `to_dict`, `case_id`, `geometry_path` |
| `RunConfig`, `GeometryConfig`, `SpeciesConfig`, `PhysicsConfig`, `ElectricFieldConfig`, `ResolutionConfig`, `SolverConfig`, `ParallelConfig`, `ConvergenceConfig`, `OutputConfig`, `ScanConfig`, `ScanAxis` | the frozen dataclasses of each table ({doc}`inputs`) |
| `CaseValidationError` | raised with `path`, `value`, `expected`, `correction` attributes |
| `case_json_schema()` | the JSON Schema that `dkx schema --format json` prints |
| `run(case, out=None, emit=None)` | solve a case; returns `Result` |
| `Result`, `RESULT_SCHEMA_VERSION` | the result object and its file schema ({doc}`outputs`) |

A `Case` is a frozen dataclass, so a variant is made with
`dataclasses.replace` and gets its own `case_id`:

```python
from dataclasses import replace

finer = replace(case, resolution=replace(case.resolution, theta=15, pitch=16))
```

Keyword overrides in the SFINCS spelling (`dkx.run(case, Ntheta=25)`) are
refused for a native case; they apply only to namelist input.

`emit` receives progress lines; `None` uses `print` when `run.progress` is
true. Refinement studies are in `dkx.workflows.converge.converge_case` and
scans in `dkx.workflows.scan.run_scan` ({doc}`convergence`,
{doc}`scans_and_parallelism`).

## SFINCS namelists

`dkx.run` also accepts a namelist path, an in-memory `SfincsInput`, or
SFINCS parameters as keywords, and dispatches on `RHSMode`:

```python
run = dkx.run("input.namelist")                       # RHSMode=1 -> ProfileRun
run = dkx.run("input.namelist", Ntheta=25, Nxi=40)    # the deck with overrides
run = dkx.run(geometryScheme=1, Ntheta=15, ...)       # no file at all
run = dkx.run("input.namelist", out="sfincsOutput.h5")
print(run.moments["FSABjHat"])
```

| name | purpose |
|---|---|
| `load_sfincs_input(path, validate=True)` | read and validate a deck into `SfincsInput` |
| `SfincsInput` | typed deck; `from_params(**names)`, `to_namelist()`, `write(path)`, `.raw` |
| `run_profile(deck, ...)` | `RHSMode = 1`; returns `ProfileRun` (`state_vector`, `moments`, `solve_result`, `operator`, `input`, `output_path`) |
| `run_transport_matrix(deck, ...)` | `RHSMode = 2/3`; returns `TransportRun` (`transport_matrix`, `state_vectors`, `moments`, ...) |
| `run_from_namelist(deck, out_path=...)` | run a deck and write its output file, dispatching on `RHSMode` |
| `write_output(deck, path, **kwargs)` | the Python form of `dkx sfincs write-output`; accepts `equilibrium_file=` or `wout_path=` |
| `read_output(path)` | read `.h5`, `.nc` or `.npz` output into a dict |
| `run_monoenergetic_database(deck, nu_prime_grid, e_star_grid)`, `monoenergetic_database(...)` | monoenergetic coefficient database |
| `find_ambipolar_er(deck, er_bracket=...)`, `run_ambipolar_brent(deck, er_min=..., er_max=...)` | ambipolar $E_r$ for a deck (Brent) |
| `ambipolar_er(deck_or_operator)` | differentiable ambipolar $E_r$ as a JAX scalar |
| `build_impurity_plasma`, `classical_impurity_flux` | impurity and classical-flux helpers ({doc}`../physics/phi1_and_impurities`) |
| `plot(source, out)` | panel figure from a run or an output file |

The drivers take `solve_method` and `tol` for common use, or a full
`solver=dkx.SolverOptions(...)` that supersedes both:

| `SolverOptions` field | default | meaning |
|---|---|---|
| `method` | `"auto"` | `auto`, `block_tridiagonal` (structured direct), `gmres` (recycled Krylov), `direct` (sparse direct) |
| `tol`, `atol` | `1e-10`, `0.0` | relative tolerance per right-hand side, absolute floor |
| `restart` | `None` | FGMRES cycle size; `None` is the memory-aware policy |
| `recycle_dim` | `8` | recycled directions |
| `max_restarts` | `200` | outer-cycle cap; exceeding it sends `auto` to sparse direct |
| `differentiable` | `False` | wrap the solve for implicit differentiation |
| `use_preconditioner`, `preconditioner` | `True`, `None` | coarse-operator preconditioner on/off, or `coarse`, `multigrid`, `sparse`, `none` |
| `device` | `None` | JAX device or platform string |
| `memory_budget_gb` | `None` | budget for the route choice (and, with MUMPS, the process envelope) |
| `direct_backend` | `None` | `None` is SuperLU; `"mumps"` needs SOLVAX 0.25.0 or later and PyMUMPS |
| `krylov_memory_budget_gb` | `None` | memory for the FGMRES basis; `None` is a quarter of available memory |
| `keep_lowest` | `3` | Legendre blocks recovered by the memory-saving structured route; set it to `Nxi` for residual audits of the full state |
| `cores` | `None` | provenance only; threads are set with `DKX_CORES` or `--cores` before JAX starts |

A preconditioner changes iteration count, time and memory, not the answer.
{doc}`../numerics/solver_routes` describes the routes and the `auto` policy.

## Batched and differentiable scans

| name | purpose |
|---|---|
| `prepare_er_scan(case, surface_index=0)` | build geometry, grids and collisions for one surface of a native case, without solving |
| `batched_er_scan(problem, er_values, ...)` | solve many $E_r$ values on one surface in one batched, optionally multi-device, optionally differentiable call |
| `batched_surface_scan(operators, ...)` | solve a batch of flux surfaces |
| `batched_solve(op, batch_leaves, ...)` | the general batch over operator leaves |

```python
import jax, jax.numpy as jnp, dkx

case = dkx.Case.from_file("examples/05_ambipolar_profile/case.toml")
problem = dkx.prepare_er_scan(case, surface_index=1)

def bootstrap_current(er_kv_m):
    scan = dkx.batched_er_scan(problem, er_kv_m, differentiable=True, retain_full_state=True)
    return jnp.sum(scan.moments["FSABjHat"])

value, gradient = jax.jit(jax.value_and_grad(bootstrap_current))(jnp.array([-0.2, 0.0, 0.2]))
```

Prepared native problems take $E_r$ in kV/m; the returned moments are in
SFINCS normalization. {doc}`scans_and_parallelism` covers batching, memory
budgets and devices; {doc}`../numerics/differentiation` covers what the
gradients do and do not include.

## Runtime

| name | purpose |
|---|---|
| `configure(jax_x64=None)` | apply the DKX runtime environment; safe to repeat |
| `require_float64()` | raise unless JAX is in float64 |
| `runtime` | the runtime settings module |
| `initialize_distributed_runtime_from_env()` | JAX multi-host initialization from `DKX_DISTRIBUTED` and related variables |
| `__version__` | package version |

The lower-level types `GeometryState`, `GridState`, `OperatorState`,
`PreconditionerState`, `SolveInputs`, `SolverResult`, `TransportResult`,
`OutputSchema` and `BenchmarkReport` are documented in {doc}`../api` and
{doc}`../how_it_works`.

## Environment variables

Solver routes are chosen by arguments, never by the environment. The variables
below control the runtime, caches and diagnostics.

| variable | effect |
|---|---|
| `DKX_CORES` | CPU threads (`--cores`); set before JAX is imported |
| `DKX_CPU_DEVICES` | force several host CPU devices (for multi-device tests; not faster) |
| `DKX_NO_X64_SETUP` | do not enable float64 at import (float64 is still required) |
| `JAX_COMPILATION_CACHE_DIR`, `DKX_COMPILATION_CACHE_DIR`, `DKX_DISABLE_COMPILATION_CACHE` | persistent compilation cache location or opt-out |
| `DKX_DATA_DIR`, `DKX_OFFLINE`, `DKX_EQUILIBRIA_DIRS` | equilibrium data cache and search path |
| `DKX_DISTRIBUTED`, `DKX_PROCESS_ID`, `DKX_PROCESS_COUNT`, `DKX_COORDINATOR_ADDRESS`, `DKX_COORDINATOR_PORT` | multi-host initialization |
| `DKX_TRANSPORT_PARALLEL`, `DKX_TRANSPORT_PARALLEL_WORKERS` | worker processes for transport right-hand sides |
| `DKX_TIER1_MEMORY_BUDGET_GB`, `DKX_KRYLOV_MEMORY_BUDGET_GB` | defaults for `memory_budget_gb` and `krylov_memory_budget_gb` |
| `DKX_PHI1_NEWTON_TOL`, `DKX_PHI1_GMRES_TOL` | Newton and inner GMRES tolerances of the $\Phi_1$ solve (default `1e-12`) |
| `DKX_ROSENBLUTH_METHOD` | default Rosenbluth quadrature when no key or argument is given |
| `DKX_FORTRAN_STDOUT` | mirror (`1`) or silence (`0`) the SFINCS progress lines |
| `DKX_WRITE_SOLVER_DIAGNOSTICS` | add per-right-hand-side residual datasets to transport output |
| `DKX_PROFILE`, `DKX_PROFILE_DEVICE_MEM` | per-stage timing and memory sampling |
| `DKX_DEBUG` | full tracebacks from CLI errors |
