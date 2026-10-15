# Changelog

Each release lists its changes with the pull request that made them. The
highlights of 2.9.0: an exact speed-coupled structured direct route for
Fokker–Planck and Sugama decks that fit, the same elimination as the `coupled`
Krylov preconditioner that `auto` picks when memory allows
({doc}`numerics/solver_routes`), native $\Phi_1$ promoted to
`validated_limited` with a matrix-free adjoint
({doc}`physics/phi1_and_impurities`), spectral angles for monoenergetic
databases, a multithreaded BLAS default on CPU, and `dkx template` with case
files that no longer carry a `schema` key. New users start at
{doc}`tutorials/index`.

```{include} ../CHANGELOG.md
:heading-offset: 1
```
