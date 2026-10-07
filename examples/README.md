# Examples

Start with `tutorials/01_first_run.py` and walk down the ladder; each script
adds one idea to the one before it.

```console
python examples/tutorials/01_first_run.py
DKX_EXAMPLES_CI=1 python examples/tutorials/01_first_run.py   # seconds-long smoke run
```

Every tutorial has the same four parts: imports, input parameters, the run
(with progress printed as it goes), then plot, save and print a summary. Its
docstring explains the physics, the equations and how to read the output.
Outputs go to `examples/output/<script name>/` (gitignored). Where a CLI line
is shown, the case file sits beside the script with the same number.

| # | Tutorial | What it teaches | Case file / CLI | Full run |
| --- | --- | --- | --- | --- |
| 01 | [01_first_run.py](tutorials/01_first_run.py) | `dkx.Case` + `dkx.run`: tokamak, one ion, pitch-angle scattering | Python only | 14 s |
| 02 | [02_cli_case.py](tutorials/02_cli_case.py) | the same through `dkx validate / run / inspect` | `02_cli_case.toml` | 12 s |
| 03 | [03_sfincs_namelist.py](tutorials/03_sfincs_namelist.py) | an unchanged SFINCS deck, `sfincsOutput.h5` | `03_sfincs_namelist.namelist` | 6 s |
| 04 | [04_vmec_geometry.py](tutorials/04_vmec_geometry.py) | stellarator geometry from a VMEC `wout` | `04_vmec_geometry.toml` | 16 s |
| 05 | [05_boozer_geometry.py](tutorials/05_boozer_geometry.py) | Boozer `.bc` geometry; how the solver route is chosen | `05_boozer_geometry.toml` | 29 s |
| 06 | [06_fokker_planck_vs_pas.py](tutorials/06_fokker_planck_vs_pas.py) | collision operators: Fokker-Planck vs pitch-angle scattering | Python only | 30 s |
| 07 | [07_species_and_impurity.py](tutorials/07_species_and_impurity.py) | ions, electrons and a carbon impurity; radial current | Python only | 21 s |
| 08 | [08_ambipolar_er.py](tutorials/08_ambipolar_er.py) | `J_r(E_r)` scan and the ambipolar `E_r` roots | `08_ambipolar_er.toml` | 36 s |
| 09 | [09_monoenergetic.py](tutorials/09_monoenergetic.py) | `D11*`, `D31*`, `D33*` database vs collisionality | `09_monoenergetic.namelist` | 7 s |
| 10 | [10_transport_matrix.py](tutorials/10_transport_matrix.py) | RHSMode 2/3 transport matrices, Onsager symmetry | Python keywords | 32 s |
| 11 | [11_phi1.py](tutorials/11_phi1.py) | in-surface potential `Phi1` with an impurity | Python keywords | 61 s |
| 12 | [12_bootstrap_vs_redl.py](tutorials/12_bootstrap_vs_redl.py) | kinetic bootstrap current vs the Redl fit | Python keywords | 37 s |
| 13 | [13_convergence.py](tutorials/13_convergence.py) | why a small residual is not a converged answer | `13_convergence.toml` | 12 s |
| 14 | [14_gradients.py](tutorials/14_gradients.py) | `jax.grad` through the solve vs finite differences | Python only | 16 s |
| 15 | [15_optimization.py](tutorials/15_optimization.py) | gradient descent on a `\|B\|` harmonic | Python only | 24 s |

Full-run times are wall clock on a shared 36-core Linux CPU node, JAX
compilation included; smoke mode (`DKX_EXAMPLES_CI=1`) runs each in about
5-30 s and is what CI executes. The grids are sized for teaching, not for
publication: tutorial 13 shows how to check convergence before quoting a
number. `08_ambipolar_er_w7x.toml` is a production-scale case to
`dkx validate` (it needs the release-hosted W7-X equilibrium to run).

## Other folders

| Folder | What's inside |
| --- | --- |
| [`advanced/`](advanced) | heavier optimization scripts through the VMEX -> Boozer -> DKX chain, and the `scan-er` promotion audit |
| [`data/`](data) | small shared input decks |
| [`sfincs_examples/`](sfincs_examples) | vendored upstream SFINCS v3 decks, kept for parity and benchmark audits |

`workflow_catalog.json` is the machine-readable version of this index:

```console
python examples/list_workflows.py --list-topics
python examples/list_workflows.py --search "ambipolar"
```
