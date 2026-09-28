# Numerics

DKX solves one large, structured linear system per flux surface and right-hand
side: the radially local drift-kinetic equation of {doc}`../physics/drift_kinetic_equation`,
discretized on a tensor grid in speed, pitch angle and the two flux-surface
angles, bordered by the density, energy and gauge constraints. With
$\Phi_1$ active the system becomes nonlinear and is solved by Newton iteration
over the same linear machinery. These pages describe how the system is built,
which solver runs it, and how derivatives are taken through it.

| Page | Contents |
| --- | --- |
| {doc}`discretization` | Unknowns, angular stencils, the Legendre pitch basis, the Landreman–Ernst speed grid, the $N_\xi(x)$ ramp, the bordered block structure |
| {doc}`solver_routes` | The `auto` policy and its three routes: structured direct block elimination, recycled Krylov, sparse direct referee |
| {doc}`krylov_and_preconditioners` | GCROT/FGMRES, the memory-aware restart, the coarse preconditioner family and why multigrid does not replace it |
| {doc}`factor_reuse` | Returning and passing back factorizations; transposed solves; neighbouring operators |
| {doc}`differentiation` | Implicit-function-theorem adjoints, what is differentiable, measured gradient accuracy and cost |
| {doc}`compilation_and_parallelism` | Compile time against steady state, batched solves, multiple devices, CPU threads, where the GPU helps |

Timing tables for whole cases live in {doc}`../benchmarks/performance`; the
reasoning behind the main algorithmic choices is collected in
{doc}`../design_decisions`.

```{toctree}
:maxdepth: 1

discretization
solver_routes
krylov_and_preconditioners
factor_reuse
differentiation
compilation_and_parallelism
```
