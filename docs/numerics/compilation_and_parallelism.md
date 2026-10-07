# Compilation and parallelism

DKX runs on one node: a multi-core CPU or the node's local GPUs. It gets its
throughput from compiled kernels, from batching independent solves with
`jax.vmap`, and from the structured solver routes underneath. A single linear
system always runs whole on one device; there is no domain decomposition of one
solve across devices or hosts, which is the main architectural difference from
SFINCS, which distributes one solve over MPI ranks. Timing tables for whole
cases are in {doc}`../benchmarks/performance`; the user-facing options are in
{doc}`../user_guide/scans_and_parallelism`.

## Compile time and steady state

JAX compiles each distinct program shape once per process. The first call pays
tracing and XLA compilation; later calls with the same shapes reuse the
executable.

```{figure} ../_static/figures/transport_compile_runtime_cache_2x2.png
:alt: Compile estimate against warm steady-state solve time for four reference transport cases.
:width: 95%

Per case, compile estimate = cold first call − warm first call; steady solve is
the mean of repeated warm calls. Data in
`docs/_static/figures/transport_compile_runtime_cache_2x2.json`.
```

On the four tiny monoenergetic pitch-angle-scattering decks of that figure
(`tests/ref/monoenergetic_PAS_tiny_scheme{1,11,12,5_filtered}.input.namelist`)
the compile estimate is 1.2 to 1.4 s and the warm steady solve 0.049 to
0.063 s: at small sizes compilation dominates a single call by a factor of
twenty or more. Two consequences follow.

- **Benchmark warm.** Report repeated warm timings for steady-state speed, and
  the cold first call separately. Set `JAX_COMPILATION_CACHE_DIR` to a
  persistent directory to reuse compiled kernels across processes;
  `dkx.runtime.configure()` sets up the compilation cache.
- **Scans and optimizers run at steady-state speed** once compiled, because
  neighbouring operators have the same shapes. The structured route's
  `Tier1Solver` is a pytree: physical coefficients are arguments, not
  compile-time constants, so an $E_r$ scan point or an optimizer step reuses the
  executable.

### Eager calls compile themselves

A caller that does not wrap an objective in `jax.jit` dispatches every JAX
primitive separately. On the 16,230-unknown structured deck of
{doc}`differentiation` the gradient holds 13,698 primitives, and executed that
way it cost 1.76 to 1.79 times the compiled gradient, with the eager primal 3.5
to 4 times its compiled time. The structured route therefore compiles its
band assembly and elimination internally, and a differentiable structured solve
is one compiled `custom_linear_solve` over every right-hand-side column, with
the refined substitution and the residual guard inside it. Measured on loaded
hosts (Xeon W-2295 and an Apple-silicon laptop, `tools/benchmarks/derivative_cost.py`):

```{list-table}
:header-rows: 1

* - Ratio (process CPU time, median)
  - self-compiling route
* - eager primal / compiled primal
  - 1.18 – 1.30
* - eager gradient / compiled gradient
  - 1.27 – 1.38
* - compiled gradient / compiled primal
  - 1.13 – 1.20
```

The fixed eager overhead on a `5 x 5 x 6 x 4` deck, where compute is negligible,
is 0.017 s for a primal and 0.10 s for a gradient. The first `jax.jit` of a whole
objective costs 12 to 22 s on the same host. The recycled Krylov route and the
ambipolar root have not been measured this way.

Row generators of the coarse preconditioner are built inside the jitted
application from traced leaves. A generator that closed over the
$N_\theta N_\zeta$ streaming matrices would make them static constants of the
lowering; on a W7-X deck that captured 15.52 GB of constants and ended in an
out-of-memory kill. `tests/test_coarse_precond_constants.py` pins the traced
form.

## Batching independent solves

$E_r$ scans, surface sweeps, monoenergetic databases and optimizer populations
are embarrassingly parallel: each point is its own solve on a shared
discretization that differs in a few physics leaves. `dkx.batch` wraps
`jax.vmap` over the varying `KineticOperator` leaves:

```python
from dkx import batched_er_scan, batched_surface_scan

result = batched_er_scan(problem, er_values, devices="auto")
result = batched_surface_scan(operators)
```

Both return a `BatchedSolveResult` with stacked moments, accept
`differentiable=True` to stay inside a `jax.grad` chain, and report original
absolute and relative residuals plus a per-element `algebraic_converged` flag
recomputed from the operator, under JIT and sharding alike. Because `vmap`
amortizes per-solve dispatch, batching beats a serial Python loop on CPU: about
9.5× for an $E_r$ scan and 6.4× for a surface scan
(`tools/benchmarks/batched_scan.py`).

### Memory-budgeted chunks

The batch runs in `jax.lax.map` chunks sized from the per-solve footprint of
the route `auto` actually takes (`dkx.solve.auto_solve_peak_memory_bytes`) and
the device or host memory budget. A solve that routes to the truncated
structured kernel is charged its truncated working set, not the full-band peak
it never allocates. On the production HSX deck that charge is 1.68 GB against a
53.06 GB full-band charge (measured process peak about 1.16 GB); at a 19.2 GB
budget the chunk count rises from 1 to 11, and a mid-deck 8-point
`batched_er_scan` runs in one chunk at 749 MB instead of three chunks at
2,084 MB. `memory_budget_gb` overrides the budget for both the chunk planner and
each element's route decision; `max_batch` caps the chunk. The budget is an
estimate, not an allocation limit, and reverse mode can retain residuals across
chunks, so measure a gradient's peak separately.

## Multiple devices

`devices="auto"` splits a batch across every local device of the selected
backend with `jax.shard_map`; each device runs the same memory-budgeted local
map on its shard, inside `jax.jit` and `jax.grad`. Padding repeats the final
valid case to equalize shards and is trimmed before objectives and derivatives.
Fewer than two devices, or fewer cases than devices, uses the single-device
path. `tests/test_batch.py` checks the actual addressable shards, single-device
agreement, uneven batches, JIT, and physical-current gradients on two forced CPU
devices.

A bounded probe on two RTX A4000s (eight two-species pitch-angle-scattering
cases, `7x7` angles, eight pitch modes, three speed nodes, chunk size two)
measured synchronized warm medians over ten calls: forward 44.1 ms on one GPU and
27.7 ms on two; value and gradient 71.8 ms and 43.4 ms. Compilation and first
execution took 4.7 to 8.4 s. This teaching-size probe does not qualify
production scaling.

For multi-host device pools, JAX distributed initialization is opt-in
(`DKX_DISTRIBUTED`, `DKX_PROCESS_ID`, `DKX_PROCESS_COUNT`,
`DKX_COORDINATOR_ADDRESS`, `DKX_COORDINATOR_PORT`, or the matching CLI flags).
Independent transport right-hand sides can be spread over worker processes with
`DKX_TRANSPORT_PARALLEL` and `--transport-workers`.

## Batching inside one structured solve

The truncated structured kernel eliminates $B = N_\mathrm{species}N_x$
independent chains. `solve(subsystem_batch=...)` sets how many run at once; any
width gives identical per-chain arithmetic. The `"auto"` default is
backend-aware:

- **CPU: width 1.** XLA:CPU runs the batch axis of the LAPACK factor and solve
  calls serially per element, so a wider sweep adds memory and cache pressure
  without parallelism. On the 336,610-unknown mid HSX deck at 8 threads, the
  ramped deck took 10.3 s at width 1 against 11.4 s at width 2, and the
  uniform-`Nxi` variant 16.6 s at width 1 against 20.5 s at width 10.
- **Accelerators: the widest width whose modeled footprint fits the budget**
  (`dkx.solve.tier1_truncated_subsystem_width`), because batching raises device
  occupancy while the budget bounds the working set.

## CPU threads

XLA sizes its host threadpool once, when the CPU backend initializes, so thread
control must be set before JAX is imported. `dkx --cores N` and `DKX_CORES=N`
pin the solver threadpool to `N` threads (applied as `NPROC`, the variable XLA
reads); `DKX_CORES=0` lets XLA size it. With neither `DKX_CORES` nor `NPROC`
set, XLA gets `min(8, cores)` threads and the host BLAS `min(4, cores)`, counting
the cores the process may run on (`taskset` aware, not `os.cpu_count()`).

The clamp is measured. On a 36-core workstation the mid HSX deck's warm
structured solve takes 9.7 s at 1 thread, 7.8 s at 2, 5.6 s at 4 and 4.87 s at
8 (the optimum, 1.99×), then 12.2 s at 16, 56.6 s at 32 and 29.3 s at 36. The
operator build stays near 8 s at every count, so the inversion is XLA fork-join
overhead over the sequential Legendre sweep. Set `--cores` to about 4 to 8 on
many-core hosts, not to `nproc`.

When `DKX_CORES` or `NPROC` is set, and whenever host devices are forced, the
BLAS pools default to one thread (`dkx.runtime`); explicit `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` always win. The unset default
gives BLAS several threads because XLA's sequential LAPACK custom calls
(`getrf`, `trsm`) inside the structured elimination otherwise run on one core:
on a Xeon pinned to four cores the warm W7-X monoenergetic solve went from
2.55 s to 1.67 s and NCSX full FP from 5.2 s to 5.0 s, with the sparse direct
route and a 14-core laptop unchanged. One BLAS thread remains right for batched
kernels. JAX's CPU LAPACK kernels
already run a batch across XLA's threadpool, and a multithreaded BLAS inside
each element oversubscribes it: on a 36-thread Xeon W-2295 a batched
$777\times777$ `lu_factor` took 284 to 756 ms per matrix with eight BLAS threads
and 3.4 to 6.2 ms with one, and the NCSX `(21, 37, 61, 8)` coarse factorization
at `DKX_CORES=4` took 93 s with four BLAS threads and 33 s with one, at identical
iterations and moments. `DKX_CPU_DEVICES` forces several host devices for
multi-device tests; they share one threadpool, so it is not a performance knob.

## Where the GPU helps

The structured direct route is FP64-compute-bound on a GPU, and its truncated
working set is small, so multi-million-unknown solves fit a 16 GB card.
Warm structured solves of HSX-family decks on one RTX A4000 (12.56 GB usable
device budget, device peak from JAX memory statistics, every residual between
`1e-13` and `1e-15`):

```{figure} ../_static/figures/gpu_anatomy_memory.png
:alt: Left, RTX A4000 device-peak memory against unknown count and the 12.56 GB budget; right, mid-deck warm CPU solve time against pinned core count.
:width: 92%

Left: single-GPU memory ladder. Right: 36-core CPU thread scaling on the mid
deck (336,610 unknowns), bottoming at 8 cores. Regenerate with
`tools/benchmarks/gpu_anatomy_figure.py`.
```

```{list-table}
:header-rows: 1

* - Unknowns
  - Device peak (GB)
  - Warm solve (s)
* - 336,610
  - 0.18
  - 3.7
* - 1,275,010
  - 0.61
  - 25.6
* - 2,025,010
  - 1.42
  - 85.3
* - 2,525,010
  - 2.21
  - 157.6
```

The full-band charge for the largest row is about 208 GB. The first measured
out-of-memory point is the next rung, whose truncated estimate is 26.25 GB.

Serial, dispatch-bound paths gain least: the recycled Krylov iterations, the
$\Phi_1$ Newton solve and the ambipolar root. On the same host the GPU beat the
36-core CPU on every structured direct warm solve measured down to 6.5k unknowns
and every preconditioned Krylov warm solve down to 2.8k unknowns; the one CPU win
was the small unpreconditioned Krylov loop of a 4.5k-unknown $\Phi_1$ Newton
solve (0.048 s CPU against 0.159 s GPU). Batched work widens the GPU's lead.

## Parallel runs as a cross-check

Parallel paths call the same matrix-free operators as the serial path, so
outputs agree up to floating-point reduction order, and a batched or sharded run
is an independent check of a serial one. Parallel execution does not by itself
establish SFINCS parity or resolution convergence.
