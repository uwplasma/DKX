# Benchmarks

This section collects every measured comparison DKX makes: against analytic
limits of the drift-kinetic equation, against SFINCS Fortran v3 on identical
decks, against independent monoenergetic codes, and against its own runtime and
memory budget. Each number on these pages names the script, test or checked
artifact it comes from, so it can be regenerated.

| Page | What it answers |
| --- | --- |
| {doc}`analytic_limits` | Does the discretization reproduce known asymptotic results (Spitzer–Härm, Shaing–Callen, Pfirsch–Schlüter, Onsager symmetry, conservation)? |
| {doc}`sfincs` | Does DKX agree with SFINCS v3 on the same deck, and where does the reference itself fall short? |
| {doc}`cross_code` | Do the monoenergetic coefficients agree with MONKES, YANCC and the ICNTS benchmark set? |
| {doc}`performance` | How fast, how much memory, CPU against GPU, and which solver route to expect. |
| {doc}`validation_matrix` | Which validation entries support a release claim, and at what evidence level. |

Two rules hold across the section. A small residual is not a converged answer:
convergence in resolution is measured separately ({doc}`../user_guide/convergence`).
And a timing is only compared with another timing from the same host, the same
commit, and the same warm or cold state.

```{toctree}
:maxdepth: 1

analytic_limits
sfincs
cross_code
performance
validation_matrix
```
