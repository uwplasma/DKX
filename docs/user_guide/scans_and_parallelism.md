# Scans and parallelism

Most neoclassical work is many independent solves: a range of $E_r$, a set of
flux surfaces, a collisionality sweep, a resolution ladder. DKX runs on one
node (a multicore CPU or the node's GPUs) and gets its throughput from
running those independent solves together, not from splitting one linear
system across machines. There are three levels:

| level | tool | parallelism |
|---|---|---|
| declarative scan of a case | `[scan]` + `dkx scan` | one derived case after another, resumable |
| batched solves in Python | `batched_er_scan`, `batched_surface_scan`, `batched_solve` | `jax.vmap` in memory-bounded chunks, optionally across devices, optionally differentiable |
| inside one solve | CPU threads (`--cores`), subsystem batching | XLA thread pool; independent Legendre chains |

## Declarative scans

A `[scan]` table turns one case into many ({doc}`inputs`):

```toml
[scan]
combine = "cartesian"     # every combination; "zipped" walks the axes together
resume = true
output = "outputs/er_scan.nc"
max_cases = 10000

[[scan.axis]]
path = "electric_field.value_kV_m"
values = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]

[[scan.axis]]
path = "species[deuterium].temperature_scale"
values = [0.8, 1.0, 1.2]
```

```console
dkx validate case.toml       # prints the expanded case count and the limit
dkx scan case.toml           # runs 21 derived cases, writes one result
dkx plot outputs/er_scan.nc  # observables against the scan axis
```

Axis paths are limited to `electric_field.value_kV_m`, the four
`resolution` sizes, `solver.relative_tolerance`, `solver.memory_fraction`,
and `species[NAME].density_scale` / `temperature_scale`, which multiply that
species' whole profile. The count is checked against `max_cases` before
anything runs.

Each derived case is an ordinary case with its own `case_id`. The scan solves
them in sequence and writes one result with a `case` dimension holding each
case's ID, status and axis values, and the largest absolute particle flux,
heat flux and parallel current over its surfaces and species ({doc}`outputs`).

- **Failures do not discard work.** A point that raises is recorded as
  `failed: <reason>`, the scan continues, the file is written, and `dkx scan`
  exits 1.
- **Resume is keyed by content.** With `resume = true`, a point whose
  `case_id` is already in the output file is not solved again. Editing the case
  changes the IDs of the affected points, so they rerun instead of reusing
  rows computed for different physics. `--no-resume` reruns everything.

The scan output path is used as given (relative to the working directory),
unlike `[output].file`, which is resolved beside the case file.

## Batched solves

`dkx.batch` wraps `jax.vmap` over solves that share one discretization and
differ only in a few physics leaves:

```python
import dkx, jax.numpy as jnp

case = dkx.Case.from_file("examples/tutorials/08_ambipolar_er.toml")
problem = dkx.prepare_er_scan(case, surface_index=1)     # geometry, grids, collisions once

scan = dkx.batched_er_scan(problem, jnp.linspace(-5.0, 5.0, 21), devices="auto")
jr = scan.radial_current            # J_r = sum_s Z_s Gamma_s per E_r (normalized)
ok = scan.algebraic_converged       # one boolean per E_r
```

`prepare_er_scan` builds everything that does not depend on $E_r$ from a native
case without solving; the prepared problem takes $E_r$ in kV/m. It supports the
native domain (DKES trajectories, no $\Phi_1$) and holds geometry, profiles and
collision coefficients fixed: prepare again after changing the case.
`batched_surface_scan(operators)` solves a set of surfaces, and
`batched_solve(op, batch_leaves)` batches arbitrary operator leaves.

The returned `BatchedSolveResult` carries `states`, `moments` (SFINCS
normalization), `radial_current`, the requested `method` and the
`executed_method` per element, the chunking actually used (`chunk_size`,
`n_chunks`), and the admission evidence:

| field | meaning |
|---|---|
| `residual_norms` | $\lVert Ax-b\rVert$ recomputed from the original operator and the returned state |
| `relative_residual_norms` | the same divided by $\lVert b\rVert$ (0 for a zero drive with zero residual, otherwise infinite) |
| `algebraic_converged` | finite state, drive and residual, and relative residual within the tolerance; independent of the solver's own success flag |

`algebraic_converged` is algebraic admission only; it does not certify grid
resolution or physics. By default the memory-bounded structured route returns
only the low-order Legendre blocks with a zero tail. Its moments can be
accurate while that state fails the full equation, so pass
`retain_full_state=True` whenever the states themselves are used or the full
residual matters; chunk sizing then includes the larger working set.

`differentiable=True` keeps the batch inside `jax.grad` and `jax.jit`
({doc}`../numerics/differentiation`).

### Memory budget

There are no sharding environment variables on this path. The batch runs in
`jax.lax.map` chunks sized from the per-solve footprint of the route that
`auto` will take and the available device (or host) memory. A solve that routes
to the memory-bounded structured kernel is charged its truncated working set,
not a full-band factorization it never allocates. `memory_budget_gb` overrides
the budget for both the chunk planner and each element's route choice, so a
tight budget cannot size a small chunk and then let an element pick a route
that does not fit; `max_batch` caps the chunk size. The budget is an estimate,
not an allocation limit, and reverse-mode differentiation can keep residuals
across chunks: measure a gradient's peak memory separately.

### Devices

`devices="auto"` uses every local device of the selected backend; a sequence
selects distinct devices. JAX `shard_map` runs the memory-budgeted map on each
device's share of the batch, including inside `jax.jit` and `jax.grad`. With
fewer than two devices, or fewer cases than devices, the single-device path
runs. The batch is padded by repeating its last case to equal shares and
trimmed back in order, and padding never reaches objectives or gradients.
`tests/test_batch.py` checks the shard placement, agreement with one device,
uneven batches, JIT and gradients on two forced CPU devices; forced CPU devices
share one thread pool and demonstrate correctness, not speedup. A single solve
always runs whole on one device.

### Measured gain

Because `vmap` amortizes per-solve dispatch, batching beats a Python loop even
on a CPU: about 9.5× for an $E_r$ scan and 6.4× for a surface scan
(`python tools/benchmarks/batched_scan.py` reproduces both). The larger gain is
on a GPU, where one solve runs at about CPU speed and a batch fills the device;
iterative and small single solves are slower on a GPU than on a CPU because
they are dominated by serial, dispatch-bound iterations
({doc}`../benchmarks/performance`).

## Inside one solve

### CPU threads

XLA sizes its host thread pool once, when the CPU backend starts, so thread
control must be set before JAX is imported. The CLI does it:

```console
dkx --cores 4 run case.toml        # or: export DKX_CORES=4 before starting Python
```

`DKX_CORES=N` pins the pool to N threads (applied as `NPROC`, which XLA reads,
together with the OpenMP/OpenBLAS pools, which then get one thread); `0` lets
XLA size it. With neither `DKX_CORES` nor `NPROC` set, DKX gives XLA
`min(8, cores)` threads and the host BLAS (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) `min(4, cores)`, counting the cores
`taskset` allows: the measured XLA optimum is 4–8 threads, a full-width pool on
a many-core host is slower, and a single BLAS thread would leave the
sequential LAPACK calls of the structured elimination on one core
({doc}`../numerics/compilation_and_parallelism`). Explicit BLAS variables
always win. `DKX_CPU_DEVICES` forces several host devices for
multi-device tests; it shares the same pool and does not speed anything up.

### Subsystem batching

The memory-bounded structured direct kernel eliminates
$B = n_\text{species}\times n_x$ independent (species, speed) chains.
`dkx.solve.solve(subsystem_batch=...)` sets how many are eliminated at once;
every width computes identical per-chain arithmetic, trading memory for
concurrency. The default `"auto"` uses width 1 on CPU, where XLA runs the batch
axis of LAPACK calls serially, and the widest width that fits the memory budget
on accelerators ({doc}`../numerics/compilation_and_parallelism`).

## Namelist-route parallelism

| control | effect |
|---|---|
| `--transport-workers N` (`DKX_TRANSPORT_PARALLEL_WORKERS`) | worker processes for independent `whichRHS` solves on the output-writing path; `transport-matrix-v3` solves all drives in one multi-RHS solve instead |
| `dkx sfincs scan-er --jobs N` | parallel processes over $E_r$ points; `--index`/`--stride` split a scan across a job array |
| `--distributed` with `--process-id`, `--process-count`, `--coordinator-address`, `--coordinator-port` | JAX multi-host initialization |

## Relation to SFINCS

SFINCS v3 scales one solve across many nodes with MPI domain decomposition.
DKX targets one node and recovers scan throughput differently: batched `vmap`
over independent solves, subspace recycling across neighbouring points
({doc}`../numerics/factor_reuse`), and exact gradients in place of
finite-difference scans. Parallel paths call the same operators as the serial
path, so results agree up to floating-point reduction order and a parallel run
is an independent check of a serial one.
