# API reference

The public interface is the package top level: `import dkx` exposes the case
model, the runners, the result and the expert solve entry points below. Module
paths are stable within a major version; {doc}`how_it_works` maps every module
to its role.

## Cases and results

A {class}`dkx.Case` is the validated, immutable input
({doc}`user_guide/inputs`); {func}`dkx.run` executes it and returns a
{class}`dkx.Result` ({doc}`user_guide/outputs`).

```{eval-rst}
.. autofunction:: dkx.run.run

.. automodule:: dkx.config
   :members: Case, CaseValidationError, RunConfig, GeometryConfig, SpeciesConfig, PhysicsConfig, ElectricFieldConfig, ResolutionConfig, SolverConfig, ParallelConfig, ConvergenceConfig, OutputConfig, ScanAxis, ScanConfig, case_json_schema

.. automodule:: dkx.result
   :members: Result
```

## SFINCS-namelist runs, scans and the solve facade

{mod}`dkx.api` is the facade behind the namelist route
({doc}`user_guide/sfincs_namelist`), the batched and differentiable scans
({doc}`user_guide/scans_and_parallelism`, {doc}`numerics/differentiation`) and
the typed solver options ({doc}`numerics/solver_routes`).

```{eval-rst}
.. automodule:: dkx.api
   :members:
```

## Optimization objectives

Bootstrap-current terms for VMEX optimizations
({doc}`tutorials/bootstrap_gradients_optimization`).

```{eval-rst}
.. automodule:: dkx.bootstrap
   :members: KineticBootstrapMismatch, KineticBootstrapCurrent
```

## Namelist reader

```{eval-rst}
.. automodule:: dkx.namelist
   :members:
```

## Expert operator interface

For staged solves, `dkx.drift_kinetic.KineticOperator` is a JAX pytree.
Discrete dimensions, model switches and the `n_xi_for_x` pitch-truncation
layout are static compilation keys; physical coefficient arrays are dynamic
leaves. A change of the active pitch layout needs a new trace even when the
rectangular state shape is unchanged; operators with the same layout and
changed coefficients reuse the compiled executable.

Rebuild the operator consistently when density, temperature or geometry
changes: replacing only `n_hat` or `t_hat` on an existing operator does not
rebuild its collision coefficients. For full Fokker–Planck density scans at
fixed temperatures, masses, charges and speed grid,
`make_fokker_planck_v3_phi1_operator` stores unit-density collision kernels,
and `at_uniform_density(n_hats, n_xi=...)` assembles the operator with JAX
tensor operations, so density JVPs and VJPs need no new Rosenbluth integrals:

```python
from dataclasses import replace
from dkx.solve import solve

refreshed = replace(
    operator,
    n_hat=new_density,
    fp=kernels.at_uniform_density(new_density, n_xi=operator.n_xi),
)
result = solve(refreshed, refreshed.rhs(), differentiable=True)  # rebuild the drive too
```

`kernels` must be built with the operator's collision and grid settings. The
refresh does not differentiate temperature or change the Coulomb logarithm.

| Module | Role |
| --- | --- |
| `dkx.drift_kinetic` | `KineticOperator`, the SFINCS v3 drift-kinetic operator |
| `dkx.collisions` | pitch-angle scattering and linearized Fokker–Planck operators |
| `dkx.solve` | route selection and the structured, sparse and Krylov solvers |
| `dkx.moments` | fluxes, flows, `FSABjHat` and transport matrices |
| `dkx.er`, `dkx.ambipolar` | ambipolar root search and branch evidence |
| `dkx.phi1` | $\Phi_1$ quasineutrality Newton solve |
| `dkx.monoenergetic`, `dkx.variational`, `dkx.shaing_callen` | reduced models ({doc}`physics/reduced_models`) |
| `dkx.writer`, `dkx.io` | `sfincsOutput.h5`, NetCDF and NPZ writers and readers |
