# Changelog

Each release lists its changes with the pull request that made them. The
highlights of 2.7.0: a memory-aware Krylov restart that grows to 1,000 vectors
within a memory budget ({doc}`numerics/krylov_and_preconditioners`), a
structured direct route that compiles itself for callers that do not `jit`
({doc}`numerics/compilation_and_parallelism`), and a drift-kinetic
bootstrap-current objective for VMEX optimization
({doc}`tutorials/vmex_optimization`).

```{include} ../CHANGELOG.md
:heading-offset: 1
```
