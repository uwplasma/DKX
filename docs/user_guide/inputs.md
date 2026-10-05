# Case files

A DKX **case** is one immutable description of a calculation: the geometry,
the plasma profile, the physics model, the radial electric field, the
phase-space resolution, and the solver and output settings. It is written in
TOML by people and in JSON by programs; both formats carry exactly the same
fields and both pass through one validation boundary,
`dkx.Case.from_mapping` (`src/dkx/config.py`). `dkx.Case.from_file` reads a
`.toml` or `.json` file and calls it.

## The case model

- **No version key needed.** A case without one is read as the current format
  (version 1). An optional `format_version = 1` pins it; the legacy key
  `schema = 1` is still accepted. Any other value is refused; no migration is
  defined.
- **Unknown keys are errors.** A misspelled field is refused with the list of
  accepted names, at every level of nesting.
- **Precise errors.** A `dkx.CaseValidationError` names the field path (for
  example `species[1].density_m3`), the supplied value, the expected form and a
  correction.
- **Deterministic identity.** `case.case_id` is the SHA-256 of the normalized
  content, serialized with sorted keys. It does not depend on key order, on
  TOML versus JSON, or on where the file lives: the source path is kept only to
  resolve relative geometry paths and is excluded from the ID. Optional fields
  left at their defaults (`pitch_speed_ramp = 1`, `pitch_modes_by_speed`,
  `retain_legendre_tail = false`, `search_strategy = "uniform"`,
  `seed_brackets_kV_m`, `coulomb_logarithm = 17.0`) are dropped from the hashed
  content, so adding them explicitly at their default does not change the ID.
- **Physical units.** Field names carry their unit where one applies:
  `density_m3`, `temperature_keV`, `mass_amu`, `value_kV_m`, `search_kV_m`.
  The conversion to SFINCS normalized quantities is fixed by the reference set
  $\bar n = 10^{20}$ m$^{-3}$, $\bar T = 1$ keV, $\bar m = m_p$,
  $\bar B = 1$ T, $\bar R = 1$ m ({doc}`../physics/normalizations`).

Two commands support writing a case:

```console
dkx schema > case.toml                   # every accepted field, commented
dkx schema --format json > case.schema.json   # JSON Schema (draft 2020-12) for editors
dkx validate case.toml                   # schema + executor preflight, no solve
```

The template printed by `dkx schema` names a VMEC file and enables sharding and
convergence refinement, so it is a reference to edit down, not a file that runs
unchanged. `dkx validate` runs the schema check and then the executor's own
preflight, so a case it accepts will not be refused at run time for an
unsupported option. It does not open the geometry file.

## A complete example

```toml
name = "analytic_ambipolar_profile"

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

[electric_field]
mode = "ambipolar"
search_kV_m = [-5.0, 5.0]
search_points = 5
root_tolerance_kV_m = 0.05

[resolution]
theta = 5
zeta = 5
pitch = 8
speed = 4

[solver]
relative_tolerance = 1.0e-8
```

This is `examples/05_ambipolar_profile/case.toml` with defaulted fields
omitted. The tables `[run]`, `[geometry]`, `[[species]]`, `[physics]`,
`[electric_field]`, `[resolution]` and `[solver]` are required, even when every
field inside them takes its default; `[parallel]`, `[convergence]`, `[output]`
and `[scan]` are optional.

## Top level

| field | type | default | meaning and rule |
|---|---|---|---|
| `format_version` | integer | `1` | optional; legacy name `schema` |
| `name` | string | required | non-empty after trimming; recorded in the result |

## `[run]`

| field | type | default | allowed | meaning and rule |
|---|---|---|---|---|
| `workflow` | string | required | `profile`, `ambipolar_profile`, `transport_matrix`, `monoenergetic` | what to compute; the native executor runs only the first two (see "What the native executor runs" below) |
| `precision` | string | `"float64"` | `float64` | the only accepted value |
| `device` | string | `"auto"` | any non-empty name | `auto` lets JAX choose; any other value is passed to the solver as the device |
| `progress` | boolean | `true` | | print one line per surface while solving |

## `[geometry]`

| field | type | default | allowed | meaning and rule |
|---|---|---|---|---|
| `format` | string | required | `analytic`, `vmec`, `boozer` | source of the magnetic field |
| `file` | string | required | non-empty | for `analytic`, a built-in model name; otherwise a path to a VMEC `wout_*.nc` or a Boozer `.bc` file, resolved relative to the case file's directory |
| `surfaces` | array of numbers | required | each in $[0, 1]$, unique | normalized toroidal flux $\psi_N$ of each flux surface |

The built-in analytic models are `tokamak`, `lhd_standard`, `lhd_inward` and
`w7x_standard` (hyphens and case are ignored). They are the SFINCS
`geometryScheme` 1–4 equilibria at their default parameters; a case cannot
change their parameters.

For VMEC input the equilibrium is interpolated radially to each requested
surface. The Boozer reader detects the six-column stellarator-symmetric and the
ten-column non-symmetric SFINCS `.bc` conventions. The file is read once per
profile and its SHA-256 is recorded in the result (`geometry_sha256`).

Execution adds two rules the schema does not: at least **two** surfaces, and
the surfaces must be **strictly increasing and above zero** (the magnetic axis
has a singular radial Jacobian).

## `[[species]]`

One table per species; at least one is required.

| field | type | unit | default | rule |
|---|---|---|---|---|
| `name` | string | | required | non-empty, unique among species |
| `charge` | number | $e$ | required | nonzero; `-1` for electrons |
| `mass_amu` | number | u (atomic mass unit) | required | positive |
| `density_m3` | array of numbers | m$^{-3}$ | required | one positive value per surface |
| `temperature_keV` | array of numbers | keV | required | one positive value per surface |

The profiles give values, not gradients. The executor computes
$dn/d\hat r$ and $dT/d\hat r$ with `numpy.gradient` over the surfaces
(second-order at the ends when there are at least three surfaces), which is
why two surfaces are the minimum. Place surfaces closely enough that this
finite difference represents the profile you mean.

## `[physics]`

| field | type | default | allowed | meaning and rule |
|---|---|---|---|---|
| `model` | string | `"full_local"` | `full_local` | radially local $\delta f$ drift-kinetic model |
| `collisions` | string | `"linearized_fokker_planck"` | `linearized_fokker_planck`, `pitch_angle_scattering` | full linearized Fokker–Planck operator (SFINCS `collisionOperator = 0`) or pitch-angle scattering (`collisionOperator = 1`) |
| `magnetic_drifts` | string | `"dkes"` | `dkes`, `full` | trajectory model; only `dkes` runs |
| `phi1` | string | `"off"` | `off`, `kinetic`, `full` | poloidal/toroidal variation of the electrostatic potential; only `off` runs |
| `coulomb_logarithm` | number | `17.0` | $5 \le \ln\Lambda \le 30$ | the normalized collisionality is proportional to it; this is how a case expresses a SFINCS `nu_n` override |

`dkes` means the DKES trajectory model: the $E\times B$ drift divided by
$\langle B^2\rangle$, no radial-electric-field terms in $\dot x$ or $\dot\xi$,
and no tangential magnetic drifts ({doc}`../physics/drives_and_rhs_modes`).
The constraint scheme that removes the collision operator's null space is
chosen from the collision operator (1 for Fokker–Planck, 2 for pitch-angle
scattering) and cannot be set in a case.

## `[electric_field]`

| field | type | unit | default | rule |
|---|---|---|---|---|
| `mode` | string | | required | `prescribed` or `ambipolar`; must be `ambipolar` for `workflow = "ambipolar_profile"` and `prescribed` for `profile` |
| `value_kV_m` | number | kV/m | none | required and finite when `mode = "prescribed"`; the same $E_r$ on every surface |
| `search_kV_m` | `[min, max]` | kV/m | none | required when `mode = "ambipolar"`; finite with `min < max` |
| `find_all_roots` | boolean | | `true` | refine every sign-changing bracket, not only one |
| `continue_branches` | boolean | | `true` | follow the selected root branch from surface to surface |
| `search_points` | integer | | `9` | $\ge 3$; coarse $E_r$ samples per surface |
| `root_tolerance_kV_m` | number | kV/m | `1.0e-3` | positive; bracket width at which a root is accepted |
| `max_root_iterations` | integer | | `20` | $\ge 1$; bisection steps per bracket |
| `search_strategy` | string | | `"uniform"` | `uniform` or `seeded_brackets` |
| `seed_brackets_kV_m` | array (per surface) of arrays of `[left, right]` | kV/m | none | only with `seeded_brackets` |

The electric field is converted to the SFINCS normalized `ErHat` with the
pinned 1 keV, 1 m reference set; for that set the conversion factor is one
({doc}`../physics/electric_field`).

`search_strategy = "seeded_brackets"` promotes brackets found by an earlier,
independently reviewed discovery run at a more expensive resolution. It is
accepted only when all of these hold:

- `run.workflow = "ambipolar_profile"` and `mode = "ambipolar"`;
- `seed_brackets_kV_m` has exactly one non-empty array per surface;
- every bracket is increasing, lies inside `search_kV_m`, and the brackets of a
  surface are sorted and non-overlapping;
- `find_all_roots = true` and `convergence.enabled = false`.

With `search_strategy = "uniform"`, `seed_brackets_kV_m` must be omitted.
A seeded run examines only the given intervals; crossings outside them are not
excluded, and the result says so (`ambipolar_search_scope =
"explicit_seeded_intervals_only"` and a warning).

## `[resolution]`

| field | type | default | rule | meaning |
|---|---|---|---|---|
| `theta` | integer | required | $\ge 1$ | poloidal grid points $N_\theta$ |
| `zeta` | integer | required | $\ge 1$ | toroidal grid points per field period $N_\zeta$; `1` for axisymmetric cases |
| `pitch` | integer | required | $\ge 1$ | Legendre pitch-angle modes $N_\xi$ |
| `speed` | integer | required | $\ge 1$ | speed-grid nodes $N_x$ |
| `pitch_speed_ramp` | integer | `1` | `0`, `1`, `2` | SFINCS `Nxi_for_x_option`: how many pitch modes each speed node keeps |
| `pitch_modes_by_speed` | array of integers | none | see below | explicit active pitch-mode count per speed node |

`pitch_speed_ramp` values:

| value | rule |
|---|---|
| `0` | keep the declared `pitch` at every speed node |
| `1` | linear ramp in speed (default) |
| `2` | quadratic ramp |

Because raising `pitch` under a ramp changes the truncation at several speed
nodes at once, a resolution study should record the ramp and the active counts,
not only the maximum. The result records both
(`metadata["phase_space"]["active_pitch_modes_by_speed"]` and its sum).

`pitch_modes_by_speed` replaces the ramp with one count per speed node. It must
have exactly `speed` entries, each an integer from 4 through `pitch`,
nondecreasing, with the last equal to `pitch`, and `pitch_speed_ramp` must be
left at `1`. It changes which Legendre modes are active inside the same
operator shape; it is a diagnostic control, not by itself a convergence claim.

The other discretization choices are fixed on the native route: the
Landreman–Ernst speed grid (`xGridScheme = 5`, $x_{\max} = 5$), fourth-order
centered angular differences (scheme 2), and `NL = min(4, pitch)` Legendre
modes in the Rosenbluth potentials ({doc}`../numerics/discretization`).

## `[solver]`

| field | type | default | allowed | meaning and rule |
|---|---|---|---|---|
| `method` | string | `"auto"` | `auto`, `structured_direct`, `recycled_krylov`, `sparse_direct_referee` | linear-solver route |
| `relative_tolerance` | number | `1.0e-10` | $0 < \epsilon \le 1$ | a returned state is accepted only if $\lVert Ax-b\rVert \le \epsilon\lVert b\rVert$, recomputed from the original operator |
| `memory_fraction` | number | `0.75` | $0 < f \le 1$ | fraction of total host memory the solve may budget |
| `reuse` | string | `"auto"` | `auto`, `on`, `off` | reuse policy; only `auto` runs |

The routes are described in {doc}`../numerics/solver_routes`.
`structured_direct` eliminates the Legendre blocks in one direct sweep,
`recycled_krylov` iterates with a coarse preconditioner and a reused search
subspace, and `sparse_direct_referee` factors the assembled operator as an
independent cross-check. `auto` chooses from the operator's structure and the
memory budget. With `auto`, an ambipolar evaluation that misses the residual
target is retried once with a bounded GMRES solve; an explicitly named method
is never changed and fails instead.

## `[parallel]`

| field | type | default | allowed | meaning and rule |
|---|---|---|---|---|
| `strategy` | string | `"auto"` | `auto`, `serial`, `batch` | how independent solves are scheduled |
| `shard` | array of strings | `[]` | unique values from `surface`, `electric_field`, `species` | axes to split across devices |

The native executor solves surfaces in sequence and refuses
`strategy = "batch"` or a non-empty `shard`. Within an ambipolar surface the
coarse $E_r$ samples are already solved as one memory-bounded batch. Batched and
multi-device scans are reached through the Python API
({doc}`scans_and_parallelism`).

## `[convergence]`

| field | type | default | meaning and rule |
|---|---|---|---|
| `enabled` | boolean | `false` | adaptive refinement of the ambipolar $E_r$ search |
| `observables` | array of strings | `[]` | unique names; with `enabled`, drawn from `electric_field`, `particle_flux`, `heat_flux`, `parallel_current`, `bootstrap_current`, `radial_current` |
| `relative_tolerance` | number | `0.02` | $0 < \epsilon \le 1$ |
| `max_refinements` | integer | `3` | $\ge 0$; number of midpoint-insertion levels |
| `retain_legendre_tail` | boolean | `false` | ambipolar only; accepted for compatibility (complete states already carry the tail diagnostic) |

This table controls the **electric-field** search, not the phase-space grid:
`enabled = true` is accepted only with `workflow = "ambipolar_profile"`.
Refining `theta`, `zeta`, `pitch` and `speed` is a separate study run by
`dkx converge` ({doc}`convergence`).

With refinement enabled, each surface inserts the midpoint of every interval
for `max_refinements` levels and records, per level, the root count, root
movement, observable movement and final bracket width. `validate` and `run`
compute a conservative bound on the number of retained evaluations before
allocating anything and refuse cases above 100,000 retained evaluations per
surface (`src/dkx/workflows/ambipolar_native.py`).

## `[output]`

| field | type | default | meaning |
|---|---|---|---|
| `file` | string | `"dkx_result.nc"` | NetCDF path, relative to the case file's directory; `dkx run` writes it unless `--out` is given |
| `plots` | boolean | `true` | `dkx run` writes a `.png` summary beside the Result; the Python `dkx.run` never plots (use `Result.plot`) |

## `[scan]`

| field | type | default | meaning and rule |
|---|---|---|---|
| `combine` | string | `"cartesian"` | `cartesian` (every combination) or `zipped` (axes walked together; equal lengths required) |
| `resume` | boolean | `true` | skip points whose `case_id` is already in the output |
| `output` | string | `"dkx_scan.nc"` | combined NetCDF result |
| `max_cases` | integer | `10000` | $\ge 1$; the expanded case count may not exceed it |
| `[[scan.axis]]` | array of tables | required, $\ge 1$ | each has `path` (string) and `values` (non-empty array of numbers) |

An axis `path` must be unique and one of

- `electric_field.value_kV_m`
- `resolution.theta`, `resolution.zeta`, `resolution.pitch`, `resolution.speed`
- `solver.relative_tolerance`, `solver.memory_fraction`
- `species[NAME].density_scale` or `species[NAME].temperature_scale`, where
  `NAME` is a declared species and the value multiplies that species' whole
  profile.

A case with `[scan]` is run with `dkx scan`, not `dkx run`
({doc}`scans_and_parallelism`).

## What the native executor runs

The schema is deliberately wider than the executor, so that a case can name a
model before the native route implements it. `dkx validate` and `dkx run`
refuse these values with the field path and a correction; they are never
silently replaced:

| field | schema accepts | native executor accepts |
|---|---|---|
| `run.workflow` | `profile`, `ambipolar_profile`, `transport_matrix`, `monoenergetic` | `profile`, `ambipolar_profile` |
| `physics.magnetic_drifts` | `dkes`, `full` | `dkes` |
| `physics.phi1` | `off`, `kinetic`, `full` | `off` |
| `electric_field.mode` | `prescribed`, `ambipolar` | the one matching the workflow |
| `geometry.surfaces` | one or more, in $[0,1]$ | two or more, strictly increasing, above zero |
| `parallel.strategy`, `parallel.shard` | as tabled above | `auto`/`serial`, no shard axes |
| `solver.reuse` | `auto`, `on`, `off` | `auto` |
| `convergence.enabled` | `true`/`false` | `true` only for `ambipolar_profile` |
| `[scan]` | present | expanded by `dkx scan`, refused by `dkx run` |

For the calculations outside this set, use the SFINCS namelist route
({doc}`sfincs_namelist`):

| calculation | namelist route |
|---|---|
| Onsager transport matrix | `RHSMode = 2`; `dkx sfincs transport-matrix-v3`, `dkx.run_transport_matrix` |
| monoenergetic coefficients $D_{11}^*, D_{31}^*, D_{33}^*$ | `RHSMode = 3`; `dkx sfincs monoenergetic-database`, `dkx.run_monoenergetic_database` (`examples/04_monoenergetic_scan`) |
| full trajectories ($E_r$ terms in $\dot x$, $\dot\xi$; $B^2$ in the $E\times B$ drift) | `useDKESExBDrift`, `includeXDotTerm`, `includeElectricFieldTermInXiDot` |
| tangential magnetic drifts | `magneticDriftScheme` 1–9 |
| $\Phi_1$ and quasineutrality | `includePhi1 = .true.` |
| improved Sugama model operator | `collisionOperator = 3` |
| modified analytic equilibria, inline Boozer spectra | `geometryScheme` 1–4 parameters, `geometryScheme = 13` |

## JSON cases

A JSON case has the same structure, with `species` and `scan.axis` as arrays of
objects:

```json
{
  "name": "tokamak",
  "run": {"workflow": "profile"},
  "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.16, 0.25]},
  "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
               "density_m3": [8e19, 7e19], "temperature_keV": [1.0, 0.8]}],
  "physics": {},
  "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
  "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4},
  "solver": {}
}
```

`case.to_dict()` returns the normalized content (the form that is hashed), and
the result stores it as `metadata["canonical_case"]`.
