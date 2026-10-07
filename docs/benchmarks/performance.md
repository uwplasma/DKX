# Performance and memory

This page records how fast DKX runs, how much memory it takes, and which
solver route a deck will land on. Each table names the host and the script that
measured it. Timings from different hosts are kept in separate tables and are
not compared with each other.

What decides performance is the solver route ({doc}`../numerics/solver_routes`):

| Route | Applies to | Cost profile |
| --- | --- | --- |
| Structured direct (`block_tridiagonal`, truncated kernel) | operators block-tridiagonal in the Legendre index: pitch-angle scattering, DKES trajectories | exact block elimination; memory $O(K m^2)$ with $m = N_\theta N_\zeta$, independent of $N_\xi$ |
| Recycled Krylov (`gmres`, GCROT-recycled FGMRES) | full Fokker–Planck, tangential magnetic drifts, $E_r$ `xDot`/`xiDot` terms, $\Phi_1$ | iterations under a coarse-operator preconditioner; its dense $(N_\theta N_\zeta)$ bands dominate memory at large size |
| Sparse direct (`direct`) | any operator, up to a few $10^5$ unknowns | exact assembly from operator products, Ruiz equilibration, LU |

`method="auto"` picks among them from the operator's structure and a memory
estimate.

## Head-to-head on one production case

The benchmark case is `HSX_PASCollisions_DKESTrajectories` (RHSMode=1) at
$N_\theta = 25$, $N_\zeta = 51$, $N_\xi = 100$, $N_x = 5$: 744,610 unknowns.
Both codes ran on the same Apple M4 laptop (about 10 cores, 24 GB). The Fortran
reference is SFINCS v3 with conda PETSc 3.23 and MUMPS 5.8.2; DKX uses the
truncated structured direct Legendre elimination
(`solvax` `block_thomas_truncated_fn`, `keep_lowest=3`, exact for every
RHSMode=1 output).

```{figure} ../_static/figures/readme/tier1_hsx_runtime_memory.png
:alt: Runtime and peak memory bars for DKX and SFINCS Fortran v3 on the 744k-unknown HSX PAS case.
:width: 90%

Warm solve time and peak process RSS. Measured with
`python tools/benchmarks/tier1_hsx_head_to_head.py`; drawn by
`python tools/benchmarks/readme_figures.py`.
```

| Configuration | Warm solve [s] | Peak RSS [GB] |
| --- | ---: | ---: |
| DKX, M4 CPU, $N_\xi$-for-$x$ ramp | 27.2 | 0.93 |
| DKX, M4 CPU, uniform $N_\xi$ | 44.3 | 1.16 |
| DKX, RTX A4000 GPU | 45.0 | 1.88 (0.05 GB device buffers) |
| SFINCS v3, 1 MPI rank | 463.6 | 3.98 |
| SFINCS v3, 2 MPI ranks (best measured) | 229.5 | 2.86 |

With the ramp, DKX is 17× faster than one Fortran rank and 8.4× faster than
Fortran's best parallel configuration, at about 30% of the memory. The ramp
moves the physics outputs by at most 0.9% (electrons). The GPU time equals the
CPU time here because the Legendre elimination is serial in $L$ and the A4000
runs FP64 at 1/32 rate.

Fortran strong scaling on the same case and machine saturates at two ranks:

| MPI ranks | Solve [s] | Speed-up | Peak RSS [GB] |
| ---: | ---: | ---: | ---: |
| 1 | 463.6 | 1.00 | 3.98 |
| 2 | 229.5 | 2.02 | 2.86 |
| 4 | 240.9 | 1.92 | 2.88 |
| 8 | 270.5 | 1.71 | 1.61 |

Speed is only meaningful at matched answers. The referee tests that gate each
solver route against SFINCS output are summarized here; the full parity record
is on {doc}`sfincs`.

```{figure} ../_static/figures/readme/canonical_parity.png
:alt: Parity envelopes of DKX against SFINCS v3 for profile outputs, state vectors and transport matrices.
:width: 80%

RHSMode=1 output tables to 8e-14, structured direct state vectors to 1e-11,
RHSMode=2/3 transport matrices to 6e-13 to 9e-9
(`python tools/benchmarks/readme_figures.py`).
```

This is one case, chosen because the structured direct route applies to it.
The suite benchmark below shows where it does not.

## Cold and warm solves

A warm solve is the second and later solve of the same shapes in one process,
after JAX has traced and XLA has compiled the computation. Optimizers, $E_r$
scans and convergence ladders see warm solves; a single command-line run sees a
cold one. Apple M3 Max CPU, float64, one process per row
(`tools/benchmarks/tier1_hsx_head_to_head.py --device cpu --repeat 3 --ramp`):

| Case | Unknowns | Cold [s] | Warm [s] | Cold / warm |
| --- | ---: | ---: | ---: | ---: |
| HSX PAS reduced, $(S,X,L,T,Z) = (2,12,10,13,13)$ | 40,584 | 1.72 | 0.12 | 14× |
| HSX PAS, $25 \times 51 \times 100 \times 5$ | 744,610 | 23.6 | 20.0 | 1.18× |

Compilation costs about the same for both rows, so it dominates the small case
and disappears in the large one. Timing DKX on a toy case in a fresh process
measures XLA, not the solver. The head-to-head above does not depend on the
warm number: cold, the 744k case is 23.6 s against 463.6 s and 229.5 s.

Paths that exploit warm solves carry more than compiled code:
`dkx.er.radial_current` and the ambipolar solver thread the GCROT recycle
subspace, the last solved state as initial guess, and the built preconditioner from
one $E_r$ to the next. A scan that launches one process per point (the
`sfincsScan` compatibility driver) pays a cold solve every time.

## Compile time and steady state

```{figure} ../_static/figures/transport_compile_runtime_cache_2x2.png
:alt: Compile estimate against warm steady-state solve time for four reference transport cases.
:width: 90%

For four reference transport cases: compile estimate (cold first call minus
warm first call) against the steady warm solve time. Data in
`docs/_static/figures/transport_compile_runtime_cache_2x2.json`.
```

Set `JAX_COMPILATION_CACHE_DIR` to a persistent directory to reuse compiled
executables across processes. For benchmarking, report warm repeated timings
and the cold first call separately. Details of what is compiled, and when, are
in {doc}`../numerics/compilation_and_parallelism`.

## The whole upstream suite

Every deck in the upstream `fortran/version3/examples` suite (38 decks:
geometry schemes 1/2/4/5/11 and filtered W7-X netCDF equilibria, pitch-angle
and Fokker–Planck collisions, zero and finite $E_r$, $\Phi_1$ on and off,
tangential magnetic drifts, one to three species, 651 to 1.9M unknowns) was run
end to end through both codes with `tools/benchmarks/parity_performance_matrix.py`
and plotted with `tools/paper_benchmarks/cross_code_matrix.py`.

```{figure} ../_static/figures/paper_benchmarks/cross_code_matrix.png
:alt: Speed-up and peak memory against problem size for DKX and SFINCS Fortran v3 across the upstream suite, coloured by solver route.
:width: 95%

Warm DKX solve against the Fortran wall time, coloured by the route DKX used.
```

| Route | Faster than SFINCS | What it is |
| --- | --- | --- |
| structured direct | 9 of 9 | exact block elimination over the Legendre index |
| recycled Krylov | 7 of 23 | GCROT under the coarse-operator preconditioner |

The losses sit where block-tridiagonal structure in $L$ is broken: full
Fokker–Planck collisions, tangential magnetic drifts, the $E_r$
`xDot`/`xiDot` terms, and the $\Phi_1$ Newton iteration. Those decks go through
recycled Krylov, where the reference's factorization-based preconditioner is
usually faster.

Since this sweep, small and medium Fokker–Planck decks with DKES trajectories
take a speed-coupled structured direct route instead
({doc}`../numerics/solver_routes`); the table above is the sweep as run.

- **Memory is the weak axis.** DKX uses less memory on 3 of the 32 decks it
  completed. Below about 10k unknowns the JAX runtime floor (about 0.5 GB)
  exceeds the whole Fortran process (0.1–0.2 GB). Above about 1M unknowns the
  Krylov preconditioner's dense bands dominate.
- **Six decks did not complete** in that sweep, against 38 of 38 for the
  reference. Five exceeded memory while allocating preconditioner bands; the
  routing described under "Decks whose preconditioner bands do not fit" sends
  them to a lower-memory route. The sixth needed the LIBSTELL text form of a
  VMEC `wout`, read by `dkx.vmec_ascii`.

Physics agreement across the sweep is a median relative difference of 4.1e-6
on the shared output moments ({doc}`sfincs`).

## Suite benchmark, CPU and GPU

A second, fast-to-reproduce comparison runs the 39-case CPU/GPU example suite
against SFINCS v3 and plots every row inside a 10 s reference-runtime-window:
rows whose Fortran runtime is at least 10 s, so that process launch, file I/O
and JIT compilation do not dominate. All 39 cases stay in the parity audit;
only the plot applies the window.

```{figure} ../_static/figures/paper/dkx_fortran_suite_benchmark_summary.png
:alt: Runtime and active-memory bars for SFINCS Fortran v3 and DKX CPU and GPU, cold and warm, across the example suite.
:width: 92%

Runtime (left) and active memory (right), ordered by best warm DKX speed-up.
Fortran memory is process maximum RSS; DKX memory is profiled RSS minus the
fixed Python/JAX/XLA baseline. Reproduce with
`python tools/publication_figures/generate_fortran_suite_benchmark_summary.py`.
```

Median DKX/Fortran ratios over the plotted rows, from
`tools/publication_figures/artifacts/dkx_fortran_suite_benchmark_summary.json`:

| Ratio | CPU | GPU |
| --- | --- | --- |
| cold wall clock | 0.021x | 0.037x |
| active memory | 2.89x | 3.71x |
| process maximum RSS | 4.75x | 8.80x |

The runtime ratio is dominated by the structured direct decks; the memory
ratios are the JAX floor on small decks. The summary JSON also records which
short-reference rows are excluded and the top runtime and memory cases.

## Cross-machine time to solution

End-to-end wall time (operator build, solve, moments, output) on the
two-species production variant of the HSX PAS case (1,275,010 unknowns), best
of two, against a freshly compiled SFINCS v3 (conda PETSc 3.25 + MUMPS, MPI)
(`tools/benchmarks/time_to_solution.py`):

| Configuration | End-to-end [s] |
| --- | ---: |
| Fortran MPI, 10-core laptop, 1 rank | 350 |
| Fortran MPI, 10-core laptop, best (8 ranks) | 141 |
| DKX, laptop, one process (cold / warm) | 62 / 46 |
| Fortran MPI, 36-core workstation, 1 rank | 1163 |
| Fortran MPI, 36-core workstation, best (8 ranks) | 802 |
| Fortran MPI, 36-core workstation, 32 ranks | 1423 |
| DKX, workstation, one RTX A4000 (cold / warm) | 78 / 59 |
| DKX, workstation, one CPU process (cold / warm) | 6132 / 1998 |

One DKX process is faster than every measured Fortran configuration on the
same hardware: 3.1× the laptop's best MPI time on CPU and 13.6× the
workstation's best MPI time on its GPU. On the workstation the CPU path is
limited by the serial Legendre elimination at that machine's lower single-core
speed, so the GPU is the backend to use there.

## CPU and GPU on one host

All numbers in this section come from one workstation (36 cores, RTX A4000
16 GB, JAX 0.10.2), so CPU and GPU columns compare directly. Warm is the second
identical `solve()` in-process; reproduce with `tools/benchmarks/gpu_cpu_ladder.py`.

| Route, deck family | Unknowns | CPU warm [s] | GPU warm [s] |
| --- | ---: | ---: | ---: |
| structured direct, HSX PAS/DKES | 6,488 | 1.41 | 0.53 |
| structured direct, HSX PAS/DKES | 78,010 | 6.32 | 1.48 |
| structured direct, HSX PAS/DKES | 1,275,010 | 1,036 | 26.0–26.5 |
| recycled Krylov, W7-X FP | 2,804 | 1.38 | 0.93 |
| recycled Krylov, W7-X FP | 78,628 | 21.7 | 12.0 |

On the same host the GPU won every measured size on both routes, 2.7× to 39×
on structured direct and about 1.5× to 1.8× on recycled Krylov, so there is no
same-host crossover above the smallest deck measured. All-core CPU timings on
that host carry large run-to-run variance under load. Against a fast laptop CPU
on a different machine the picture differs: a 40,584-unknown structured solve
took 0.62–0.73 s on an Apple laptop CPU against 1.17–2.20 s on the A4000.

### GPU memory headroom

```{figure} ../_static/figures/gpu_anatomy_memory.png
:alt: Left, single RTX A4000 device-peak memory against unknown count and the 12.56 GB budget. Right, mid-deck warm CPU solve time against pinned core count.
:width: 92%

Left: device peak memory of the truncated structured direct solve on one RTX
A4000 against unknowns. Right: warm CPU solve time of the 336,610-unknown deck
against pinned cores, fastest at 8 cores. Regenerate with
`python tools/benchmarks/gpu_anatomy_figure.py`.
```

| Unknowns | Device peak [GB] | Warm solve [s] |
| ---: | ---: | ---: |
| 336,610 | 0.18 | 3.7 |
| 1,275,010 | 0.61 | 25.6 |
| 2,025,010 | 1.42 | 85.3 |
| 2,525,010 | 2.21 | 157.6 |

Device memory grows far slower than the unknown count, because the truncated
kernel only materializes the lowest Legendre blocks. The 2,525,010-unknown
solve peaks at 2.21 GB where a full-band factorization would need about
208 GB. Every solve converged to residuals between 1e-13 and 1e-15.

## Memory

- At the production resolution of the HSX case ($25 \times 115 \times 149 \times 5$,
  2,512,760 unknowns) neither a global sparse factorization in SFINCS nor one in
  DKX fits a 24 GB machine: MUMPS drove macOS swap to about 46.5 GB and was
  killed. The truncated Legendre elimination is the direct path that fits; on
  the 744k case it needs about 0.3 GB where a full-band structured
  factorization would need about 91 GB.
- The small-deck floor is the XLA runtime, not DKX data
  (`tools/benchmarks/memory_floor.py`, macOS arm64, JAX 0.11). A 111-unknown
  solve peaks at 0.531 GB; `import jax` alone is 0.112 GB, the operator build
  adds 0.126 GB and the solve 0.248 GB. Inside the process, live JAX arrays
  total under 1 MB and the Python heap peaks at 5 MB. Neither
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` nor `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
  changes the peak; `JAX_DISABLE_JIT=1` raises it to 0.558 GB.

| Unknowns | Route | Peak RSS |
| ---: | --- | ---: |
| 111 | structured direct | 0.527 GB |
| 2,804 | recycled Krylov | 0.741 GB |
| 5,208 | sparse direct | 1.031 GB |
| 143,530 | recycled Krylov | 3.979 GB |

## Decks whose preconditioner bands do not fit

The recycled Krylov route stores the coarse preconditioner as dense
$(N_\theta N_\zeta)$ bands per (species, $x$, $L$). The routing is automatic by
measured size: the dense bands are used when they fit in physical RAM
(`dkx.coarse_precond._coarse_bands_fit`); otherwise the Schur-LU factors alone
when they fit (`dkx.coarse_precond._coarse_factors_fit`); otherwise a
checkpointed route that regenerates the bands. Each transition warns with both
sizes. `DKX_TIER2_MEMORY_GUARD=off` forces the dense route, and
`DKX_COARSE_FACTOR_DTYPE=float32` halves the Schur LU.

### Not yet demonstrated at production scale

Every upstream deck this route was built for completes with it, but slowly.
Measured on a 36-core, 62 GB machine with float32 reusable factors:

| Deck | Magnetic drifts | Iterations | Wall | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| `HSX_PASCollisions_fullTrajectories` | no | 11 | 50 min | 13.57 GB |
| `HSX_FPCollisions_DKESTrajectories` | no | 46 | 69 min | 13.06 GB |
| `HSX_FPCollisions_fullTrajectories` | no | 49 | 71 min | 13.22 GB |
| `filteredW7XNetCDF_2species_magneticDrifts_noEr` | yes | 123 | 1 h 56 min | 11.66 GB |
| `filteredW7XNetCDF_2species_magneticDrifts_withEr` | yes | 120 | 2 h 17 min | 11.72 GB |

The drift decks reach these counts only because the preconditioner carries the
$L$-diagonal part of the magnetic-drift terms, which Fortran also keeps
(`preconditioner_magnetic_drifts_max_L`). Without it they took 1260 and 3384
iterations (10 h 08 min and 26 h 07 min), a 10× to 28× difference, at unchanged
residuals and 0.3–0.5 GB extra storage.

Float32 factors cost nothing on these decks, but are not free in general. On
analytic decks at $9 \times 9$ angles, $N_\xi = 16$, $N_x = 5$, where the coarse
operator is nearly exact, float32 factors raise the iteration count from 3 to
228 (pitch-angle scattering) and from 12 to 474 (full Fokker–Planck); both still
reach 1e-10. Iteration counts are only comparable on the same machine and
commit.

These wall times are hours where the structured route takes seconds. That gap
is the open item for the recycled Krylov route at production scale.

## Why the coarse chain is not truncated instead

The structured direct route saves memory by keeping only the lowest $K$
Legendre blocks of the solution. The obvious lever for the coarse
preconditioner is to factor only its leading $K$ blocks, storing $O(K m^2)$
instead of $O(N_\xi m^2)$ and reusing the factors on every Krylov application.
It does not work, and the measurement is kept so it is not repeated.

The two truncations are different operations.
`solvax.direct.block_thomas_truncated_fn` sweeps every block and truncates the
retained solution, which is why its head is exact. Factoring only the leading
$K$ blocks severs the $L \pm 1$ streaming coupling at $l = K$, and in the coarse
operator that coupling is the leading term: Schur complements propagate down
the whole chain, so every block above $K$ contributes to the $l = 0$ inverse
that carries density, flow and heat flux.

Measured on `geometryScheme4_2species_noEr` ($N_\xi = 48$,
$N_\theta N_\zeta = 247$, 10 subsystems), with every block above $K$ inverted
exactly so the ladder prices the severed coupling and nothing else. GCROT
iteration counts are deterministic and independent of machine load (reproduce
with `tools/benchmarks/tier2_coarse_truncation.py`):

```{list-table} Cutting the coarse Legendre chain at l = K
:header-rows: 1

   * - blocks kept
     - GCROT iterations to 1e-10
     - relative residual reached
   * - 48 (the whole chain)
     - 26
     - 4.9e-11
   * - 47
     - 133
     - 7.9e-11
   * - 44
     - no convergence in 300
     - 1.7e-06
   * - 36
     - no convergence in 300
     - 6.9e-02
   * - 24
     - no convergence in 300
     - 4.0e-01
   * - 3
     - no convergence in 300
     - 9.8e-01
```

Cutting one link of forty-eight costs five times the iterations; cutting four
ends convergence. `tokamak_2species_PASCollisions_withEr_fullTrajectories`
($N_\xi = 40$) has the same shape at nonzero $E_r$: 19 iterations for the whole
chain, 888 with one link cut, and no convergence in 6000 at $K = 36$. Cheaper
tails (identity, diagonal Schur complements, an exact head with an approximate
tail solve) are all worse. Dropping the $L \pm 1$ coupling everywhere
(`dkx.solve.build_coarse_preconditioner` with `drop_l_coupling=True`) is the
limit of the same family and does not converge on that deck either.

The coarse operator is cheap to simplify (self-species $x$-diagonal
collisions, no $L \pm 2$ terms, no magnetic drifts; recycled Krylov corrects
those in a few extra iterations). It is not cheap to shorten. Memory has to come
from how the chain is stored, not from how much of it is kept. Storing the Schur
factors in float32 is one such saving that survives: across four decks it costs
between zero and 26% more iterations, and every case still reaches 1e-10.

| Deck | float64 | float32 |
| --- | ---: | ---: |
| `geometryScheme4_2species_noEr` | 26 | 26 |
| `geometryScheme4_2species_withEr_fullTrajectories` | 29 | 29 |
| `filteredW7XNetCDF_2species_noEr` | 20 | 22 |
| `tokamak_2species_PASCollisions_withEr_fullTrajectories` | 19 | 24 |

## Choosing a route and a device

| Situation | Recommendation |
| --- | --- |
| Pitch-angle scattering with DKES trajectories, any size | leave `method="auto"`; it picks structured direct, the fastest route on both CPU and GPU |
| Many solves of the same operator (transport matrix, gradient, $E_r$ root) | keep one process; pass `SolveResult.factors` back on direct routes ({doc}`../numerics/factor_reuse`) |
| Full Fokker–Planck, tangential drifts, $\Phi_1$, up to a few $10^5$ unknowns | try `method="direct"`: exact, and every further right-hand side costs one back-substitution |
| Same, larger | recycled Krylov (`auto`); expect minutes to hours and budget memory for the preconditioner bands |
| Workstation with a data-center or workstation GPU | use it: the same-host GPU won every measured size |
| Laptop CPU with high single-core speed | CPU is competitive with a mid-range GPU on single solves |
| Batches over $E_r$, surfaces or species | `dkx.batched_er_scan`, `dkx.batched_surface_scan`; see {doc}`../user_guide/scans_and_parallelism` |
| Benchmarking | report cold and warm separately, on one host, with the same commit |

`DKX_PROFILE=1` emits flushed phase timings for operator construction,
preconditioner construction and the outer Krylov solve; `JAX_LOG_COMPILES=1`
logs compilation.

## Against MONKES and YANCC

This section compares speed and memory only. Agreement between the codes is
covered on {doc}`cross_code`. Each code was run on the same `booz_xform` files:
W7-X EIM at $s = 0.2$ and HSX QHS at $s = 0.25$. Both devices were run at
$\nu^* = \nu R_0/(v\iota) = 0.672, 0.0672, 0.00672$ and at
$v_E = E_r/(vB_{00}) = 0$ and $4.1\times10^{-4}$, giving six points per device.
A seventh case is the full-Fokker–Planck NCSX problem from YANCC's own
SFINCS test ($r_N = 0.5$, $E_r = -3$ kV/m), compared with YANCC only.

Each code first ran a resolution ladder $(N_\theta, N_\zeta, N_\xi)$ from
$11\times23\times32$ to $31\times63\times128$. The table uses, for each code, the
cheapest rung that comes within 3% of that code's own finest rung, as the
largest change over $D_{11}$, $D_{31}$, $D_{33}$ and the six points. The NCSX
case uses 10% because its particle flux nearly cancels. At the matched rung,
the NCSX flow and heat flux from both codes are within 3% of SFINCS's tabulated
values. All runs used the office host, a Xeon W-2295 shared with other users
(load 18–36 during the runs). Each run is a fresh process pinned to the four
least busy physical cores. The values are medians of three repeats per point,
with the order of the codes alternated between repeats. The JAX codes each had
16 XLA threads, and MONKES had 4 OpenBLAS threads. Cold is the first solve,
including tracing and compilation. Warm is the same solve repeated. For MONKES,
which has no compile step, cold is the process wall time. GPU runs used one
A4000 at low utilization.

| Case | Code | Resolution | Error vs finest | CPU cold [s] | CPU cold, cached [s] | CPU warm [s] | Peak RSS [GB] | GPU warm [s] | GPU peak [GB] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W7-X | MONKES | 15x31x48 | 0.34% | 0.45 | - | 0.31 | 0.05 | - | - |
| W7-X | DKX | 15x31x48 | 0.44% | 11.7 | 4.81 | 3.43 | 1.02 | - | - |
| W7-X | YANCC | 25x51x96 | 1.70% | 71.1 | - | 22.8 | 2.49 | - | - |
| HSX | MONKES | 19x41x64 | 1.77% | 3.24 | - | 2.96 | 0.09 | - | - |
| HSX | DKX | 19x41x64 | 1.78% | 31.0 | 13.0 | 19.5 | 1.67 | - | - |
| HSX | YANCC | 19x41x64 | 1.80% | 62.5 | - | 11.0 | 2.35 | - | - |
| NCSX full FP | DKX | 19x41x81x7 | 5.12% | 115.8 | - | 73.3 | 8.86 | 40.6 | 6.93 |
| NCSX full FP | YANCC | 19x41x81x7 | 8.57% | 178.1 | - | 59.8 | 5.73 | 14.0 | 1.38 |

The W7-X and HSX rows use DKX's default spectral angles and were measured at
load 12–21. The NCSX rows and the GPU columns come from an earlier run at the
same settings; the monoenergetic GPU runs were not repeated because both GPUs
were busy. "Cold, cached" is DKX's first solve with its persistent
compilation cache on, which is the default outside this harness.

```{figure} ../_static/figures/benchmarks/cross_code_speed.png
:alt: Wall time per solve (cold light, warm solid) and peak RSS for MONKES, DKX and YANCC on W7-X, HSX and the NCSX full-DKE case.
:width: 90%

Matched-accuracy wall time and peak RSS on four pinned cores. Run
`python tools/benchmarks/cross_code_speed.py` to regenerate the figure and the
table from `docs/_static/figures/benchmarks/cross_code_speed.json`, which also
holds every ladder and repeat. Add `measure --help` to re-measure.
```

**Where DKX loses.** On the monoenergetic problem on this CPU, MONKES is still
faster: 11× warm on W7-X and 7× on HSX, and 4–11× cold with DKX's cache on. It
also uses 18–20× less memory. The causes:

1. *Angular discretization (matched).* MONKES uses Fourier collocation in
   $\theta$ and $\zeta$. So does `monoenergetic_database`, unless the deck
   sets `thetaDerivativeScheme`/`zetaDerivativeScheme`, and DKX's ladder tracks
   MONKES's: on W7-X 0.44% at $15\times31\times48$ and $6\times10^{-5}$ one rung
   higher (MONKES 0.34% and $4\times10^{-5}$). On HSX DKX reaches 3% at
   $19\times41\times64$, the same rung as MONKES.
   With the SFINCS finite-difference angles the errors were 1.39% and 0.6% on
   W7-X and 7.7% at $19\times41\times64$ on HSX.
2. *Dense kernels on x86 (open).* At the same rung both codes do the same block
   elimination over the Legendre index, with blocks of size
   $N_\theta N_\zeta$. The remaining warm gap is the speed of those kernels on
   this host. JAX 0.11.2 takes 7.4 ms for one $465\times465$ LU and 6.6 ms to
   solve it against 465 right-hand sides; SciPy's LAPACK on the same cores takes
   2.2 ms and 3.5 ms. On an Apple M-series laptop the two are equal (1.0 ms), and
   DKX's warm W7-X solve takes 0.40 s there.
3. *Compilation and memory.* The full-band structured route
   builds each Legendre row inside the elimination scan, keeps only the Schur
   LUs and rebuilds the off-diagonal blocks in the substitution sweeps. Its
   compiled program no longer unrolls a loop over $N_\xi$. Against stacking the bands first, on the
   laptop at W7-X $15\times31\times48$, this takes the first solve without a
   cache from 8.2 s to 3.7 s and the peak RSS from 1.95 GB to 0.94 GB, for an 8%
   slower warm solve. With the default persistent cache the first solve takes 1.4 s
   there, and 4.8 s on the shared host. About 0.25 GB of DKX's footprint is the
   JAX runtime and a further 0.1 GB the first compiled operator, before any
   solve. MONKES streams its blocks in Fortran and has neither.

**Where DKX wins.** DKX's warm solve is 7× faster than YANCC's on W7-X (on HSX YANCC is 1.8× faster), and
its cold start is 2–6× faster on every case. On the GPU, DKX's warm W7-X solve
took 0.65 s, close to MONKES's CPU time. On the
full-FP case, DKX reaches a lower error than YANCC at the same grid (5.1% vs
8.6% against each code's finest rung). DKX has the faster cold start there,
but YANCC is 1.2× faster warm and 2.9× faster on the GPU, with 1.5–5× less
memory. YANCC's matrix-free multigrid avoids DKX's dense $(N_\theta N_\zeta)^2$
preconditioner bands. *Lever:* the coarse-operator preconditioner bands
dominate DKX's footprint on this route, so a matrix-free smoother at the finest
level is the memory lever there.

**Trap found on the way.** YANCC 6f399a2 deadlocks in its preconditioner setup
when XLA's CPU thread pool has 4 or fewer threads. This happens whether the
pool is set with `NPROC` or comes from `taskset` alone. Both JAX codes were
therefore given 16 threads on the same four cores.
