# User guide

DKX has two input routes and one solver underneath them.

| route | input | entry points | returns |
|---|---|---|---|
| native case | a TOML or JSON case file, or a Python mapping, in SI units | `dkx run CASE`, `dkx.run(dkx.Case.from_file(...))` | `dkx.Result`, written as NetCDF4 |
| SFINCS namelist | a SFINCS v3 `input.namelist`, dimensionless | `dkx input.namelist`, `dkx sfincs ...`, `dkx.run("input.namelist")` | `ProfileRun` / `TransportRun`, written as `sfincsOutput.h5` (or `.nc`, `.npz`) |

The native case states a physical profile (densities in m$^{-3}$, temperatures
in keV, $E_r$ in kV/m) on several flux surfaces and is validated field by
field. It covers radial profiles at a prescribed $E_r$ and ambipolar $E_r$
profiles, with DKES trajectories, without $\Phi_1$ or with kinetic $\Phi_1$ on profile cases. The namelist route reads
SFINCS decks unchanged and reaches the rest of the model: transport matrices
(`RHSMode = 2`), monoenergetic coefficients (`RHSMode = 3`), full trajectories,
tangential magnetic drifts and the remaining $\Phi_1$ options. `dkx convert` translates a deck into a
case when the physics fits the native route and refuses, naming the key, when
it does not.

```python
import dkx

result = dkx.run(dkx.Case.from_file("case.toml"))   # native: dkx.Result
run = dkx.run("input.namelist")                     # namelist: ProfileRun or TransportRun
```

```{toctree}
:maxdepth: 1

inputs
sfincs_namelist
outputs
cli
python_api
scans_and_parallelism
convergence
```
