# Learn DKX: the tutorial track

Eight pages take you from installing DKX to bootstrap-current gradients, in
Python and on the command line with TOML. Every page shows the code, the case
file, the output printed by running it, and what to look at next. Each step is a
script in `examples/tutorials/`; run them from the repository root. The grids
are chosen to run in seconds to a minute, **not to be converged**: page 5
measures how far they are.

| Page | Scripts | You learn |
| --- | --- | --- |
| {doc}`01_first_run` | `01`, `02` | install, what a case is, `dkx.run`, `dkx run`, the first plot |
| {doc}`02_reading_results` | `01`, `02`, `03` | fluxes, current, units, the certificate, what "converged" means, SFINCS namelists |
| {doc}`03_geometry_species` | `04`, `05`, `07` | tokamak, VMEC, Boozer, W7-X/HSX; several species and an impurity |
| {doc}`04_collisions_solvers` | `06` | PAS against Fokker-Planck, solver routes, judging a run |
| {doc}`05_scans_resolution` | `13`, `08` | `dkx converge`, resolution ladders, $E_r$ scans |
| {doc}`06_ambipolar_er` | `08` | the root search, ion and electron roots, failures |
| {doc}`07_transport_matrices` | `09`, `10` | monoenergetic $D_{11}$, $D_{31}$, $D_{33}$; RHSMode 2 and 3 |
| {doc}`08_phi1_bootstrap_gradients` | `11`, `12`, `14`, `15` | $\Phi_1$, bootstrap against Redl, `jax.grad`, optimization |

Every script is flat: imports, input parameters, the run with progress printed,
then plots, saved results and a printed summary in `examples/output/<script>/`.
To change a run, edit a constant. `DKX_EXAMPLES_CI=1` shrinks each to a
seconds-long smoke run. `python docs/figures/publish_tutorial_figures.py`
copies the figures the scripts wrote into the docs.

Case files beside the scripts: `02_cli_case.toml`, `04_vmec_geometry.toml`,
`05_boozer_geometry.toml`, `08_ambipolar_er.toml`, `08_ambipolar_er_w7x.toml`,
`13_convergence.toml`, and the namelists `03_sfincs_namelist.namelist` and
`09_monoenergetic.namelist`.

## Beyond the track

| Folder | Contents |
| --- | --- |
| `examples/advanced/` | VMEX QA/QH/QI bootstrap optimizations with a kinetic row, differentiable-kinetic optimizations and the scripts that audit an optimized design |
| `examples/sfincs_examples/` | every deck of the upstream SFINCS v3 example suite, used for parity and benchmarks |
| `examples/data/` | small shared input files |

`python examples/list_workflows.py --search "VMEC geometry"` searches
`examples/workflow_catalog.json`. Parity checks and benchmarks live in `tools/`.

```{toctree}
:maxdepth: 1

01_first_run
02_reading_results
03_geometry_species
04_collisions_solvers
05_scans_resolution
06_ambipolar_er
07_transport_matrices
08_phi1_bootstrap_gradients
```
