# Learn DKX: the tutorial track

The track goes from installing DKX to putting a kinetic bootstrap current into
a stellarator optimization, in four parts. Each step shows the code or
command, the output it printed when these pages were written, what is
happening physically, and the equations behind it. Every step has a script in
`examples/tutorials/`, numbered in the order of the track; run them from the
repository root.

| Part | Steps | Scripts | Key equations |
| --- | --- | --- | --- |
| {doc}`first_steps` | install; first Python run; the CLI with a `case.toml` and with a SFINCS namelist; convergence | `01`–`03`, `13` | the drift-kinetic equation, the drives, the flux moments, the normalizations |
| {doc}`building_a_case` | analytic tokamak, VMEC and Boozer geometry; PAS, Fokker–Planck and Sugama collisions; species and impurities | `04`–`07` | $\mathbf{b}\cdot\nabla$, the radial drift, the Lorentz operator |
| {doc}`transport_physics` | $E_r$ scans and ambipolar roots; monoenergetic $D_{11}$, $D_{31}$, $D_{33}$; the thermal transport matrix; $\Phi_1$ | `08`–`11` | the ambipolarity condition, the monoenergetic equation, quasineutrality |
| {doc}`bootstrap_gradients_optimization` | bootstrap current against Redl; `jax.grad` through the solve; a shape derivative; VMEX optimization and `vmex --neoclassical` | `12`, `14`, `15`, `advanced/` | $\langle j_\parallel B\rangle$, the Sauter–Redl formula, the adjoint |

## The scripts

| Script | Teaches |
| --- | --- |
| `01_first_run.py` | `dkx.Case`, `dkx.run`, SI outputs, the certificate, NetCDF and a plot |
| `02_cli_case.py`, `02_cli_case.toml` | the same case through `dkx run case.toml` |
| `03_sfincs_namelist.py`, `03_sfincs_namelist.namelist` | a SFINCS v3 deck run unchanged, `sfincsOutput.h5` |
| `04_vmec_geometry.py`, `.toml` | a VMEC `wout` equilibrium |
| `05_boozer_geometry.py`, `.toml` | a Boozer `.bc` file, and a route chosen from the operator's structure |
| `06_fokker_planck_vs_pas.py` | full Fokker–Planck against pitch-angle scattering |
| `07_species_and_impurity.py` | ions, electrons and a carbon impurity |
| `08_ambipolar_er.py`, `.toml`, `08_ambipolar_er_w7x.toml` | every $E_r$ root, classified |
| `09_monoenergetic.py`, `.namelist` | $D_{11}^*$, $D_{31}^*$, $D_{33}^*$ against collisionality |
| `10_transport_matrix.py` | `RHSMode = 2` and `3` transport matrices |
| `11_phi1.py` | $\Phi_1$ with an impurity |
| `12_bootstrap_vs_redl.py` | kinetic $\langle j\cdot B\rangle$ against the Redl formula |
| `13_convergence.py`, `.toml` | refining every axis: a small residual is not a converged answer |
| `14_gradients.py` | `jax.grad` through the solve, checked against central differences |
| `15_optimization.py` | a shape derivative and five steps of gradient descent |

Every script is flat: imports, the input parameters, the run with progress
printed, then plots, saved results and a printed summary in
`examples/output/<script>/`. There is no argument parsing; to change a run,
edit a constant. `DKX_EXAMPLES_CI=1` shrinks each to a seconds-long smoke run.

The tutorial resolutions are chosen to run in seconds, **not to be
converged**: script `13` measures how far they are, and the answer is "far".
Refine before quoting a number ({doc}`../user_guide/convergence`).

## Beyond the track

| Folder | Contents |
| --- | --- |
| `examples/advanced/` | VMEX QA/QH/QI bootstrap optimizations with a kinetic row, differentiable-kinetic optimizations (`optimize_QA_bootstrap.py`, `optimize_electron_root.py`, `optimize_impurity_screening.py`, ...) and the promotion scripts that audit an optimized design with full scans |
| `examples/sfincs_examples/` | every deck of the upstream SFINCS v3 example suite, used for parity and benchmarks |
| `examples/data/` | small shared input files |

`python examples/list_workflows.py --search "VMEC geometry"` searches
`examples/workflow_catalog.json`, which lists each workflow with its command
and runtime. Parity checks, benchmarks and figure generators live in `tools/`.

```{toctree}
:maxdepth: 1

first_steps
building_a_case
transport_physics
bootstrap_gradients_optimization
```
