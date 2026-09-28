# Solving for the radial electric field

In a stellarator the radial electric field $E_r$ is not a free input. It is set
by ambipolarity: no net charge may leave the flux surface,

$$
J_r(E_r) = \sum_s Z_s e\,\Gamma_s(E_r) = 0 .
$$

$J_r(E_r)$ can have one root or three: an ion root at negative or small $E_r$,
an electron root at large positive $E_r$, and an unstable root between them.
This tutorial runs `examples/05_ambipolar_profile`, which searches every
surface for every root, and shows how to read what it found. The physics is in
{doc}`../physics/electric_field`.

In a tokamak intrinsic ambipolarity holds: the neoclassical particle fluxes
balance for any $E_r$, so ambipolarity cannot determine it (Helander & Simakov
2008). Use the ambipolar workflow on stellarators.

## The case

Two changes turn a profile case into an ambipolar one: the workflow and the
electric-field mode. From `examples/05_ambipolar_profile/case.toml`:

```toml
[run]
workflow = "ambipolar_profile"

[geometry]
format = "analytic"
file = "w7x_standard"
surfaces = [0.09, 0.16]

[[species]]
name = "deuterium"
charge = 1
mass_amu = 2.014
density_m3 = [8.0e19, 7.0e19]
temperature_keV = [1.0, 0.8]

[[species]]
name = "electron"
charge = -1
mass_amu = 0.000548579909
density_m3 = [8.0e19, 7.0e19]
temperature_keV = [1.0, 0.8]

[physics]
collisions = "linearized_fokker_planck"
magnetic_drifts = "dkes"

[electric_field]
mode = "ambipolar"
search_kV_m = [-5.0, 5.0]
find_all_roots = true
continue_branches = true
search_points = 5
root_tolerance_kV_m = 0.05
max_root_iterations = 20
```

Ambipolarity needs at least two species of opposite charge, quasineutral on
every surface. The `[electric_field]` keys control the search:

| Key | Default | Meaning |
| --- | --- | --- |
| `search_kV_m` | none | interval in which to sample $J_r(E_r)$ for sign changes |
| `search_points` | 9 | number of samples across that interval |
| `find_all_roots` | `true` | refine every bracketed sign change, not only the first |
| `continue_branches` | `true` | carry root branches from one surface to the next |
| `root_tolerance_kV_m` | 1e-3 | width to which each bracket is refined |
| `max_root_iterations` | 20 | refinement iterations per root |
| `search_strategy` | `"uniform"` | `"uniform"` sampling or `"seeded_brackets"` |
| `seed_brackets_kV_m` | none | per-surface brackets for `"seeded_brackets"` |

A root is only found if its sign change falls between two samples, so a coarse
`search_points` can step over a pair of close roots. The example uses 5 to run
fast.

## Run it

```console
python examples/05_ambipolar_profile/run.py
dkx run examples/05_ambipolar_profile/case.toml --out examples/output/05_ambipolar_profile/result.nc
dkx roots examples/output/05_ambipolar_profile/result.nc
```

`dkx roots` prints the root table and branch events stored in the result
(`--format json` for machine-readable output).

## Read the roots

```python
import numpy as np
import dkx

result = dkx.run(dkx.Case.from_file("examples/05_ambipolar_profile/case.toml"))
a = result.arrays
roots = np.asarray(a["ambipolar_root_kV_m"])          # every root, per surface
kinds = np.asarray(a["ambipolar_root_type"])          # ion, electron or unstable
counts = np.asarray(a["ambipolar_root_count"])        # roots found per surface
slopes = np.asarray(a["ambipolar_root_slope_A_m2_per_kV_m"])
selected = np.asarray(a["selected_ambipolar_root"])   # index of the selected root
field = np.asarray(a["electric_field_kV_m"])          # the selected E_r per surface
```

The type of each root follows from the sign of $dJ_r/dE_r$ at the root: roots
where the current rises through zero are stable, the one in between is
unstable. When a surface has several stable roots, DKX selects one and records
the reason in the result; the fluxes and currents reported for that surface are
the ones at the selected field. The other roots are kept, not discarded, so a
reader can check a selection or pick a different branch.

```{figure} ../_static/figures/docs/ambipolar_er_roots.png
:alt: Radial current against radial electric field with ion, unstable and electron roots.
:width: 70%

Radial current against $E_r$ with its roots.
```

## At benchmark scale

```{figure} ../_static/figures/paper_benchmarks/w7x_ambipolar_er.png
:alt: Radial current against electric field on W7-X with every sampled evaluation and the classified roots.
:width: 85%

$J_r(E_r)$ on W7-X with every sampled evaluation and the resulting ion,
unstable and electron roots (`tools/paper_benchmarks/w7x_ambipolar_er.py`).
```

`examples/05_ambipolar_profile/w7x_case.toml` is a production-scale W7-X case;
`dkx validate` checks it without running it. A five-surface W7-X
ambipolar profile of this kind is recorded in
`validation/native_ambipolar_profile_v1.json`, and its resolution ladder shows
that roots and fluxes still move under refinement at the resolutions tried
(outcome `refinement_exhausted`, {doc}`../benchmarks/validation_matrix`). Treat
$E_r$ as a resolution-sensitive output and check it with `dkx converge`.

## From Python, one surface at a time

For a single surface, `dkx.find_ambipolar_er` runs a Brent root search on a
SFINCS deck or a prepared problem, threading warm starts and the recycle
subspace from one evaluation to the next. `dkx.ambipolar_er` returns a root that
is differentiable: its derivative with respect to any input comes from the
implicit function theorem, $dE_r/dp = -(\partial J_r/\partial p)/(\partial J_r/\partial E_r)$,
checked against central differences in `tests/test_er.py`. For a scan of
$J_r$ over many fields in one compiled call, use `dkx.prepare_er_scan` and
`dkx.batched_er_scan` ({doc}`gradients`, {doc}`../user_guide/scans_and_parallelism`).

The SFINCS-compatible route is also available: `dkx sfincs ambipolar` runs an
in-process Brent solve from an `input.namelist`, and `dkx sfincs scan-er`
followed by `dkx sfincs ambipolar-solve` reproduces the upstream scan-directory
workflow.
