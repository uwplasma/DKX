# Examples

Every example is a flat script: imports, input parameters as plain variables
with a one-line comment each, the run, then plots, saved results and a printed
summary. There is no `main()`, no command-line parsing and no environment
switch; to change a run, edit a variable. The docstring says what the script
shows and roughly how long it takes. Outputs go to `examples/output/<script>/`
(gitignored).

```console
python examples/tutorials/01_first_run.py
```

## Tutorials

Fast: each runs in seconds to a minute on a CPU with default solver settings.
The grids are sized for teaching, not for publication; tutorial 13 shows how to
check convergence before quoting a number.

| Topic | Script | What it shows | Time |
| --- | --- | --- | --- |
| first run | [01_first_run.py](tutorials/01_first_run.py) | `dkx.Case` + `dkx.run`: tokamak, one ion, pitch-angle scattering | 33 s |
| CLI and TOML | [02_cli_case.py](tutorials/02_cli_case.py) | the same through `dkx validate / run / inspect` (`02_cli_case.toml`) | 19 s |
| SFINCS namelist | [03_sfincs_namelist.py](tutorials/03_sfincs_namelist.py) | an unchanged SFINCS deck, `sfincsOutput.h5` | 19 s |
| VMEC geometry | [04_vmec_geometry.py](tutorials/04_vmec_geometry.py) | stellarator geometry from a VMEC `wout` | 44 s |
| Boozer geometry | [05_boozer_geometry.py](tutorials/05_boozer_geometry.py) | Boozer `.bc` geometry; how the solver route is chosen | 141 s |
| collision operators | [06_collision_operators.py](tutorials/06_collision_operators.py) | pitch-angle scattering, Fokker-Planck, Sugama | 102 s |
| species, impurity | [07_species_and_impurity.py](tutorials/07_species_and_impurity.py) | ions, electrons, a carbon impurity; radial current | 46 s |
| E_r scan, ambipolar roots | [08_ambipolar_er.py](tutorials/08_ambipolar_er.py) | `J_r(E_r)` scan and the ambipolar roots (`08_ambipolar_er.toml`) | 146 s |
| monoenergetic (DKES-like) | [09_monoenergetic.py](tutorials/09_monoenergetic.py) | `D11*`, `D31*`, `D33*` vs collisionality | 33 s |
| transport matrix | [10_transport_matrix.py](tutorials/10_transport_matrix.py) | RHSMode 2 and 3 matrices, Onsager symmetry | 71 s |
| Phi1 | [11_phi1.py](tutorials/11_phi1.py) | in-surface potential with an impurity | 133 s |
| bootstrap current | [12_bootstrap_vs_redl.py](tutorials/12_bootstrap_vs_redl.py) | kinetic bootstrap current vs the Redl fit | 156 s |
| convergence | [13_convergence.py](tutorials/13_convergence.py) | why a small residual is not a converged answer | 328 s |
| gradients | [14_gradients.py](tutorials/14_gradients.py) | `jax.grad` through the solve vs finite differences | 68 s |
| optimization | [15_optimization.py](tutorials/15_optimization.py) | gradient descent on a `\|B\|` harmonic | 137 s |
| full trajectories | [16_full_trajectories.py](tutorials/16_full_trajectories.py) | full vs DKES trajectories at finite `E_r` | 152 s |
| VMEX geometry | [17_vmex_geometry.py](tutorials/17_vmex_geometry.py) | solve a VMEX equilibrium, then its transport (needs `vmex`) | 68 s |
| GPU | [18_gpu.py](tutorials/18_gpu.py) | device placement, CPU vs GPU timing | 29 s |
| plotting | [19_plotting.py](tutorials/19_plotting.py) | `dkx.plot` panels and your own figure | 29 s |

Times are wall clock on the shared 36-core Linux office node with a load
average of 90 to 150 from other jobs, compilation included, so they are upper
bounds; a quiet machine is several times faster.

## Advanced

[`advanced/`](advanced) holds the research-scale scripts: shape optimization for
impurity screening (needs only DKX, a few minutes), the VMEX QA bootstrap-current
optimization with a kinetic row, and the same row at the ambipolar `E_r` (tens of
minutes, need `vmex` and `booz_xform_jax`). See [`advanced/README.md`](advanced/README.md).

## Other folders

| Folder | What's inside |
| --- | --- |
| [`data/`](data) | small shared input decks |
| [`sfincs_examples/`](sfincs_examples) | vendored upstream SFINCS v3 decks, kept for parity and benchmark audits |

`08_ambipolar_er_w7x.toml` is a production-scale case to `dkx validate`; it
needs the release-hosted W7-X equilibrium to run.
