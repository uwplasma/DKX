# Physics

DKX solves the radially local, linearized, steady-state drift-kinetic equation on one flux
surface: the model of SFINCS version 3 (Landreman, Smith, Mollén & Helander, Phys. Plasmas
21, 042503, 2014). The unknown is the non-Maxwellian part $f_{s1}$ of each species
distribution. Particle and heat fluxes, parallel flows, the bootstrap current and transport
coefficients are velocity-space and flux-surface moments of that solution.

These pages state the equations the code assembles, the normalizations, and which options
each input route accepts.

| Page | Content |
| --- | --- |
| {doc}`drift_kinetic_equation` | $f_0+f_1$ split, streaming, mirror, $E\times B$ and magnetic drifts, trajectory models, constraints |
| {doc}`normalizations` | Reference scales, $\Delta$, $\alpha$, $\nu_n$, radial coordinates, SI conversion |
| {doc}`collisions` | Pitch-angle scattering, linearized Fokker-Planck, improved Sugama model operator |
| {doc}`drives_and_rhs_modes` | Thermodynamic and inductive drives, `RHSMode` 1/2/3, moments, transport matrix |
| {doc}`electric_field` | Radial electric field, ambipolarity, root finding and classification |
| {doc}`phi1_and_impurities` | $\Phi_1$ quasineutrality, poloidal variation in collisions, classical impurity transport |
| {doc}`geometry` | Analytic schemes, VMEC `wout`, Boozer `.bc`, the differentiable Fourier path |
| {doc}`reduced_models` | Monoenergetic database, variational bounds, Shaing-Callen limit, $1/\nu$ surrogate, momentum correction, validity flags |

## Two input routes

DKX accepts two kinds of input, and they do not expose the same physics.

Native case
: A TOML/JSON case (see {doc}`../user_guide/inputs`) executed by `dkx.run(case)`. It
  covers the profile and ambipolar-profile workflows with a deliberately narrow physics
  slice, checked in `src/dkx/execution.py` (`_validate_native_slice`).

SFINCS namelist (expert path)
: A SFINCS v3 `input.namelist` (see {doc}`../user_guide/sfincs_namelist`) built by
  `dkx.drift_kinetic.kinetic_operator_from_namelist`. It reaches every term of the v3
  operator that DKX implements.

```{list-table} Physics reachable from each route
:header-rows: 1

* - Physics
  - Native case
  - SFINCS namelist
* - Streaming and mirror
  - yes
  - yes
* - $E\times B$ drift
  - DKES form only (`use_dkes_exb=True`)
  - DKES or full (`useDKESExBDrift`)
* - $E_r$ terms in $\dot\xi$ and $\dot x$
  - no
  - `includeElectricFieldTermInXiDot`, `includeXDotTerm`
* - Tangential magnetic drifts
  - no (`physics.magnetic_drifts = "dkes"` is the only accepted value)
  - `magneticDriftScheme` 1-9
* - Collisions
  - `linearized_fokker_planck` (default) or `pitch_angle_scattering`
  - `collisionOperator` 0, 1, 3
* - `RHSMode`
  - 1 only (the case format also lists `transport_matrix` and `monoenergetic` workflows; the executor refuses them)
  - 1, 2, 3
* - $\Phi_1$
  - no (`physics.phi1 = "off"` is the only value the executor runs)
  - `includePhi1`, `readExternalPhi1`
* - Geometry
  - `analytic` (`tokamak`, `lhd_standard`, `lhd_inward`, `w7x_standard`), `vmec`, `boozer`
  - `geometryScheme` 1, 2, 3, 4, 5, 11, 12, 13
* - Ambipolar $E_r$
  - `workflow = "ambipolar_profile"`
  - `dkx.er.find_ambipolar_er`, `dkx.er.ambipolar_er`
```

The case format (`src/dkx/config.py`) accepts `physics.magnetic_drifts = "full"` and
`physics.phi1 = "kinetic"` or `"full"`, but the executor refuses them with a
`CaseValidationError` that names the field. Use the namelist route for those models.

```{toctree}
:maxdepth: 1

drift_kinetic_equation
normalizations
collisions
drives_and_rhs_modes
electric_field
phi1_and_impurities
geometry
reduced_models
```
