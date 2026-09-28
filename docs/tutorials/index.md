# Tutorials

Each tutorial is one task worked from start to finish, built on the numbered
examples in `examples/` so every command on these pages runs as written. They
assume DKX is installed ({doc}`../getting_started/installation`) and that
commands are run from the repository root.

| Tutorial | Task | Builds on |
| --- | --- | --- |
| {doc}`first_profile` | run one native case, read its SI outputs and certificate, and check convergence | `examples/01_tokamak_profile`, `examples/06_convergence_certificate` |
| {doc}`ambipolar_er` | solve for the radial electric field from ambipolarity and read every root | `examples/05_ambipolar_profile` |
| {doc}`transport_matrix` | compute monoenergetic coefficients and the thermal transport matrix | `examples/04_monoenergetic_scan`, `examples/transport` |
| {doc}`stellarator_from_vmec` | run on a VMEC `wout` or a Boozer `.bc` file | `examples/02_vmec_stellarator`, `examples/03_boozer_stellarator` |
| {doc}`gradients` | differentiate an output with `jax.grad` and check it against finite differences | `examples/07_gradients`, `examples/autodiff` |
| {doc}`vmex_optimization` | put a kinetic transport objective into a stellarator optimization | `examples/08_vmex_optimization`, `examples/optimization` |
| {doc}`sfincs_migration` | run existing SFINCS v3 decks, convert them to native cases, and compare outputs | `examples/sfincs_examples` |

The example resolutions are chosen to run in seconds, not to be converged.
Every tutorial that produces a number says how to check it.

```{toctree}
:maxdepth: 1

first_profile
ambipolar_er
transport_matrix
stellarator_from_vmec
gradients
vmex_optimization
sfincs_migration
```
