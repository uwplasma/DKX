# SFINCS namelist input

DKX reads SFINCS v3 input decks (`input.namelist`) unchanged. A deck describes
one flux surface in dimensionless form, and it reaches every model DKX
implements: profile solves (`RHSMode = 1`), the Onsager transport matrix
(`RHSMode = 2`), monoenergetic coefficients (`RHSMode = 3`), full trajectories,
tangential magnetic drifts, $\Phi_1$, and the ambipolar $E_r$ search. Decks whose
physics fits the native route can be translated into a case with
`dkx convert` (see [Converting a deck](#converting-a-deck)).

```console
dkx input.namelist                         # solve and write sfincsOutput.h5
dkx input.namelist --out run.nc            # the suffix picks HDF5, NetCDF4 or NPZ
dkx sfincs write-output --input input.namelist --out sfincsOutput.h5
```

```python
import dkx

run = dkx.run("input.namelist")                     # ProfileRun or TransportRun, in memory
run = dkx.run("input.namelist", Ntheta=25)          # the deck with one parameter overridden
inp = dkx.load_sfincs_input("input.namelist")       # typed, validated SfincsInput
```

## How the parser reads a deck

The typed parser is `dkx.inputs.SfincsInput` (`dkx.load_sfincs_input`). Group
and key names match case-insensitively. Each typed field takes its Python type
from its default. Two behaviours matter:

- **Species arrays** accept indexed assignment (`Zs(2) = 6.0`) and are folded
  into vectors; the number of species is the number of `Zs` entries.
- **Unknown keys are kept, not typed.** A key that is not a typed field of a
  recognized group is dropped from the typed fields and retained in
  `SfincsInput.raw`; an unknown group is likewise kept in `raw`. A few
  DKX-specific keys (`RosenbluthMethod`, `SfincsMatrixThreshold`) are read from
  `raw` by the operator builder.

`SfincsInput.from_params(**flat_names)` builds a validated input without a
file, using the same Fortran names, and raises `ValueError` on an unknown name.
`SfincsInput.to_namelist()` and `.write(path)` serialize an input back to
Fortran-readable text (non-default fields only unless `include_defaults=True`);
re-parsing reproduces every typed field, the `export_f` group, the untyped keys
in `raw`, and the rank-2 `boozer_bmnc(m,n)` spectra.

Validation replicates `validateInput.F90`: `RHSMode` must be 1, 2 or 3 (the
Fortran adjoint modes 4 and 5 are replaced by `jax.grad`), `collisionOperator`
must be 0, 1 or 3, and `RHSMode = 3` applies the Fortran overrides
(pitch-angle scattering, one species, no $\Phi_1$, no $E_r$ term in $\dot\xi$,
one speed node at $x = 1$), each recorded as a warning.

## Normalization

Every namelist quantity is a ratio to a reference ("Bar") set. The SFINCS
defaults `Delta = 4.5694e-3` and `nu_n = 8.330e-3` hold together only for

$$
\bar n = 10^{20}\ \mathrm{m^{-3}},\quad \bar T = 1\ \mathrm{keV},\quad
\bar m = m_p,\quad \bar B = 1\ \mathrm{T},\quad \bar R = 1\ \mathrm{m},\quad
\ln\Lambda = 17,
$$

with $\bar v = \sqrt{2\bar T/\bar m}$. `dkx.units` holds these values and
recomputes `Delta` and `nu_n` from them; `tests/test_units.py` pins the two
against the Fortran defaults. So `nHats = 0.8` means $8\times10^{19}$ m$^{-3}$,
`THats = 1.2` means 1.2 keV, `mHats` is the mass in proton masses, and `Er` is
in kV/m. The ordering parameters are

$$
\Delta = \frac{\bar m\,\bar v}{e\,\bar B\,\bar R},\qquad
\alpha = \frac{e\,\bar\Phi}{\bar T},\qquad
\nu_n = \frac{\bar\nu\,\bar R}{\bar v}.
$$

{doc}`../physics/normalizations` derives them;
[outputs](outputs.md#sfincs-output-files) gives the conversion of the output
quantities to SI.

## `&general`

| name | default | meaning |
|---|---|---|
| `RHSMode` | `1` | 1 profile solve, 2 transport matrix ($3\times3$), 3 monoenergetic ($2\times2$) |
| `ambipolarSolve` | `.false.` | solve for the ambipolar $E_r$ |
| `ambipolarSolveOption` | `2` | root finder (2 = Brent) |
| `NEr_ambipolarSolve` | `20` | number of $E_r$ samples |
| `Er_search_tolerance_dx` / `Er_search_tolerance_f` | `1e-8` / `1e-10` | root tolerances in $E_r$ and in radial current |
| `Er_min` / `Er_max` | `-100` / `100` | $E_r$ search bracket |
| `outputFilename` | `"sfincsOutput.h5"` | output file name |
| `solveSystem` | `.true.` | `.false.` writes geometry and grids only |
| `saveMatlabOutput` / `saveMatricesAndVectorsInBinary` | `.false.` | accepted for compatibility |

## `&geometryParameters`

| name | default | meaning |
|---|---|---|
| `geometryScheme` | `1` | 1–4 analytic models; 5 VMEC `wout`; 11/12 Boozer `.bc` (symmetric / non-symmetric); 13 inline Boozer spectrum |
| `equilibriumFile` | `""` | VMEC or Boozer path (aliases `JGboozer_file`, `JGboozer_file_NonStelSym`, `fort996boozer_file`) |
| `GHat` / `IHat` / `iota` | `3.7481` / `0.0` / `0.4542` | Boozer covariant components and rotational transform (analytic schemes) |
| `B0OverBBar` | `1.0` | reference field ratio |
| `psiAHat` / `aHat` | `0.15596` / `0.5585` | edge toroidal flux and minor radius, normalized |
| `epsilon_t` / `epsilon_h` / `epsilon_antisymm` | `-0.07053` / `0.05067` / `0.0` | scheme 1 toroidal, helical and antisymmetric amplitudes |
| `helicity_l` / `helicity_n` | `2` / `10` | scheme 1 helical mode numbers |
| `helicity_antisymm_l` / `helicity_antisymm_n` | `1` / `0` | scheme 1 antisymmetric mode numbers |
| `NPeriods` | `0` | field periods (0 = from the scheme) |
| `min_Bmn_to_load` / `rippleScale` | `0.0` / `1.0` | Boozer harmonic filter and ripple scaling |
| `inputRadialCoordinate` | `3` | surface label: 0 $\hat\psi$, 1 $\psi_N$, 2 $\hat r$, 3 $r_N$ |
| `inputRadialCoordinateForGradients` | `4` | gradient coordinate: 0–3 as above, 4 = $\hat r$ gradients with $E_r$ as the field input |
| `psiHat_wish` / `psiN_wish` / `rHat_wish` / `rN_wish` | `-1` / `0.25` / `-1` / `0.5` | requested surface (`normradius_wish` aliases `rN_wish`) |
| `VMECRadialOption` / `VMEC_Nyquist_option` | `1` / `1` | VMEC radial interpolation and Nyquist convention |

Equilibrium files are searched in order: the absolute path (or a
`--equilibrium-file` / `--wout-path` override); relative to the deck's
directory; relative to the working directory; the directories in
`DKX_EQUILIBRIA_DIRS`; the bundled test and example data; and finally the
release-hosted public fixtures by basename, fetched into the `DKX_DATA_DIR`
cache unless `DKX_OFFLINE=1` ({doc}`../getting_started/installation`). With an
override, the `input.namelist` copy embedded in the output reflects the
effective file.

## `&speciesParameters`

One entry per species; all are float arrays.

| name | meaning |
|---|---|
| `Zs` / `mHats` / `nHats` / `THats` | charge, mass, density and temperature per species |
| `dNHatdpsiHats` / `dNHatdpsiNs` / `dNHatdrHats` / `dNHatdrNs` | density gradient in each radial coordinate |
| `dTHatdpsiHats` / `dTHatdpsiNs` / `dTHatdrHats` / `dTHatdrNs` | temperature gradient in each radial coordinate |
| `withAdiabatic` / `adiabaticZ` / `adiabaticMHat` / `adiabaticNHat` / `adiabaticTHat` | adiabatic species (default off; $Z=-1$, $\hat m = 5.446\times10^{-4}$, $\hat n = \hat T = 1$) |
| `withNBIspec` / `NBIspecZ` / `NBIspecNHat` | beam species (default off) |

## `&physicsParameters`

| name | default | meaning |
|---|---|---|
| `Delta` / `alpha` / `nu_n` | `4.5694e-3` / `1.0` / `8.330e-3` | ordering parameters |
| `EParallelHat` | `0.0` | inductive parallel electric field |
| `dPhiHatdpsiHat` / `dPhiHatdpsiN` / `dPhiHatdrHat` / `dPhiHatdrN` / `Er` | `0.0` | radial electric field in each coordinate |
| `collisionOperator` | `0` | 0 linearized Fokker–Planck, 1 pitch-angle scattering, 3 improved Sugama model operator |
| `constraintScheme` | `-1` | null-space constraint; −1 picks 1 for Fokker–Planck and 2 for pitch-angle scattering |
| `includeXDotTerm` | `.true.` | $E_r$ term in $\dot x$ |
| `includeElectricFieldTermInXiDot` | `.true.` | $E_r$ term in $\dot\xi$ |
| `useDKESExBDrift` | `.false.` | $E\times B$ drift with $\langle B^2\rangle$ (DKES) instead of $B^2$ |
| `include_fDivVE_term` | `.false.` | compressible $E\times B$ correction |
| `includeTemperatureEquilibrationTerm` | `.false.` | collision operator acting on the other species' Maxwellians |
| `includePhi1` | `.false.` | quasineutrality and $\Phi_1$ block (nonlinear Newton–Krylov solve) |
| `includePhi1InKineticEquation` | `.true.` | $\Phi_1$ terms in the kinetic equation |
| `includePhi1InCollisionOperator` | `.false.` | poloidal density variation in the collision operator |
| `quasineutralityOption` | `1` | 1 full Boltzmann, 2 EUTERPE adiabatic |
| `readExternalPhi1` | `.false.` | fixed external $\Phi_1$ (linear solve) |
| `nuPrime` / `EStar` | `1.0` / `0.0` | monoenergetic collisionality and electric field (`RHSMode = 3`) |
| `magneticDriftScheme` | `0` | tangential magnetic drifts, schemes 0–9 as in Fortran v3 |
| `Krook` | `0.0` | non-conserving drag term |

## `&resolutionParameters`

| name | default | meaning |
|---|---|---|
| `Ntheta` / `Nzeta` | `15` / `15` | poloidal and toroidal grid points (rounded to odd when `forceOddNthetaAndNzeta`) |
| `Nxi` | `16` | Legendre pitch modes |
| `NL` | `4` | Legendre modes in the Rosenbluth potentials |
| `Nx` | `5` | speed nodes (1 for `RHSMode = 3`) |
| `xMax` / `NxPotentialsPerVth` | `5.0` / `40.0` | speed extent (used for `xGridScheme < 5`) and potential resolution |
| `solverTolerance` | `1e-6` | relative residual target |
| `forceOddNthetaAndNzeta` | `.true.` | round angular sizes up to odd |

The Fortran convergence-scan ramp arrays (`NthetaNumRuns`, `Nx_min`, ...) have
no typed fields: they stay in `raw` and are not acted on. `dkx converge
input.namelist` is the replacement ({doc}`convergence`).

## `&otherNumericalParameters`

| name | default | meaning |
|---|---|---|
| `thetaDerivativeScheme` / `zetaDerivativeScheme` | `2` / `2` | angular finite differences; `±103`/`±104` select DKX's widened upwind stencils |
| `ExBDerivativeSchemeTheta` / `ExBDerivativeSchemeZeta` | `0` / `0` | only `0` is supported (see below) |
| `magneticDriftDerivativeScheme` | `3` | magnetic-drift upwinding; also accepts `±103`/`±104` |
| `xDotDerivativeScheme` | `0` | speed-derivative scheme |
| `xGridScheme` / `xPotentialsGridScheme` / `xGrid_k` | `5` / `2` / `0.0` | speed-grid family and weight exponent |
| `Nxi_for_x_option` | `1` | pitch modes per speed node: 0 uniform, 1 linear, 2 quadratic |
| `useIterativeLinearSolver` | `.true.` | accepted (alias `useIterativeSolver`); DKX chooses its route with `--solve-method` |
| `whichParallelSolverToFactorPreconditioner` / `PETSCPreallocationStrategy` | `1` / `1` | PETSc settings, accepted for compatibility |
| `RosenbluthMethod` | `'quadpack'` | DKX extension: `quadpack`, `hybrid` or `analytic` quadrature of the Rosenbluth potentials |
| `SfincsMatrixThreshold` | `0.0` | DKX extension: drop Fokker–Planck entries at or below this magnitude |

`RosenbluthMethod = 'quadpack'` is the Fortran v3 algorithm and the parity
route. `'hybrid'` keeps QUADPACK for $L = 0 \ldots 3$ (so the particle, energy and
momentum conservation moments are unchanged) and evaluates $L \ge 4$ in closed
form; `'analytic'` uses closed form at every $L$ and may differ from Fortran at
strict-parity level. The `DKX_ROSENBLUTH_METHOD` environment variable applies
only when neither the key nor an API argument is given.

`SfincsMatrixThreshold` reproduces SFINCS's `sparsify.F90`, which skips matrix
entries with magnitude at or below `1d-12`. When the electron thermal speed is
much larger than the ion one, the ion→electron field-particle block of the
Fokker–Planck operator falls below that level and SFINCS solves a different
discrete operator. On an HSX-like two-species deck with $T_e/T_i = 23$ the
bootstrap current differed by 12% and 19% at two surfaces; with the threshold
applied in DKX the two codes agreed to $2\times10^{-10}$ and
$4\times10^{-11}$ (SFINCS v3 `8df5453` on macOS arm64 with PETSc 3.20.2 and
MUMPS 5.6.2; DKX with SOLVAX 0.20.0 and JAX 0.11.1 on CPU; the deck is a
collaborator's equilibrium that is not checked in). `tests/test_sfincs_matrix_threshold.py`
checks that the default keeps every entry and that the threshold drops only
ion→electron entries. Set it to `1d-12` only to reproduce a SFINCS reference;
it applies to `collisionOperator = 0` without `includePhi1InCollisionOperator`
and raises otherwise. The default `0.0` is the discretization of the stated
model.

## `&preconditionerOptions`

| name | default | meaning in DKX |
|---|---|---|
| `preconditioner_species` | `1` | 1 = self-collisions only in the coarse operator |
| `preconditioner_x` / `preconditioner_x_min_L` | `1` / `0` | speed-diagonal collisions in the coarse operator |
| `preconditioner_xi` | `1` | drop the L±2 terms in the coarse operator; DKX always does, so the key changes nothing |
| `preconditioner_theta` / `preconditioner_zeta` | `0` / `0` | angular coarsening |
| `preconditioner_theta_min_L` / `preconditioner_zeta_min_L` | `0` / `0` | minimum $L$ kept per angle |
| `preconditioner_magnetic_drifts_max_L` | `2` | magnetic-drift truncation in the coarse operator |
| `reusePreconditioner` | `.true.` | reuse the factorization across right-hand sides |

These shape the coarse preconditioner of the recycled Krylov route
({doc}`../numerics/krylov_and_preconditioners`).

```{note}
``preconditioner_xi=1`` drops the **L±2** terms, not the L±1 ones. The Fortran
call sites read *"Drop the off-by-2 diagonal terms in L if this is the
preconditioner"* (`populateMatrix.F90` lines 625, 772, 872, 933, 1046). DKX
drops them unconditionally, so the key is accepted for namelist compatibility
and changes nothing.

The L±1 streaming and mirror coupling is a separate, much stronger truncation,
reachable only through the `drop_l_coupling` argument of
`dkx.solve.build_coarse_preconditioner`. It is not what `preconditioner_xi`
does. Streaming and mirror are the dominant terms and are strictly
off-diagonal in the Legendre index, so severing them leaves a preconditioner
that no longer resembles the operator: on the checked-in deck
`examples/sfincs_examples/tokamak_2species_PASCollisions_withEr_fullTrajectories`
(`Nxi = 40`) the
recycled Krylov solve converges in 19 iterations with the whole chain kept and
does not converge within 6000 iterations once four links are cut
({doc}`../numerics/krylov_and_preconditioners`).
```

## Unsupported inputs

A deck that asks for physics DKX does not solve is refused with the key, its
value, what the key would change and where in the Fortran it acts. It is never
run with the key ignored:

| key | refused value | reason |
|---|---|---|
| `includeTemperatureEquilibrationTerm` | `.true.` (with `RHSMode` ≠ 3) | DKX builds only the $f_1$ part of the collision operator, so unequal-temperature heat fluxes would be wrong |
| `ExBDerivativeSchemeTheta` / `ExBDerivativeSchemeZeta` | nonzero | DKX uses the centered angular operator for $E\times B$ advection, not the sign-upwinded one |
| `withNBIspec` | `.true.` with `includePhi1` and not `readExternalPhi1` | the beam charge density is not included in quasineutrality |
| `EParallelHatSpec_bcdatFile` | non-empty | a spatially varying inductive drive; DKX applies `EParallelHatSpec` as a flux-surface constant |
| `force0RadialCurrentInEquilibrium` | `.false.` | DKX assembles only the current-free form |
| `RHSMode` | 4, 5 | adjoint modes; use `jax.grad` ({doc}`../numerics/differentiation`) |

Other gaps against SFINCS Fortran v3:

- **MPI across nodes.** DKX runs on one node (multicore CPU or the node's GPUs);
  there is no distributed-memory solve of a single system.
- **Convergence-scan ramp arrays** are kept in `raw` and not acted on.

## Converting a deck

```console
dkx convert input.namelist case.toml            # .toml or .json by extension
dkx convert input.namelist case.toml --name w7x_r05 --force
```

`dkx convert` (`dkx.input_compat.convert_sfincs_namelist`) translates a deck
into a native case ({doc}`inputs`). The translation is real, not a rename:

- **One surface becomes three.** A deck states one surface with prescribed
  gradients; a case states a profile. The converter writes three surfaces
  spaced uniformly in $r_N$ around the deck's surface (centred when there is room,
  one-sided at the edge of the equilibrium) and profiles linear in $\hat r$,
  so `numpy.gradient` recovers the deck's gradients exactly at its surface.
- **Units become SI.** `nHats`, `THats`, `mHats` and `Er` become `density_m3`,
  `temperature_keV`, `mass_amu` and `value_kV_m`.
- **`nu_n` becomes a Coulomb logarithm**: `coulomb_logarithm = 17 * nu_n / 8.330e-3`,
  the exact inverse of how the executor scales collisionality, so the round
  trip reproduces the deck's `nu_n`.
- **Ambipolar decks** become `workflow = "ambipolar_profile"` with
  `search_kV_m = [Er_min, Er_max]`, `search_points = max(3, NEr_ambipolarSolve)`
  and `root_tolerance_kV_m = Er_search_tolerance_dx`.
- `Ntheta`, `Nzeta`, `Nxi`, `Nx`, `Nxi_for_x_option` and `solverTolerance`
  carry over to `[resolution]` and `solver.relative_tolerance`.
- A VMEC or Boozer path is written relative to the destination case file.

Conversion refuses rather than approximating. Each refusal names the namelist
key. The deck must use:

| key | required value |
|---|---|
| `RHSMode` | 1 (use `dkx sfincs transport-matrix-v3` or `monoenergetic-database` for 2 and 3) |
| `geometryScheme` | explicit; 1–4, 5, 11 or 12 (13 carries an inline spectrum a case cannot hold) |
| scheme 1–4 parameters (`epsilon_*`, `iota`, `GHat`, `helicity_*`, `psiAHat`, ...) | their defaults; a case names the analytic model only by token |
| `VMECRadialOption` | 0 (interpolate to the surface) |
| `VMEC_Nyquist_option`, `min_Bmn_to_load`, `rippleScale` | defaults |
| `Delta`, `alpha` | the reference-set values |
| `Krook`, `EParallelHat`, `EParallelHatSpec` | zero |
| `includePhi1` | `.false.` |
| `magneticDriftScheme` | 0 |
| `useDKESExBDrift`, `includeXDotTerm`, `includeElectricFieldTermInXiDot` | the DKES trajectory values, unless $E_r = 0$ where they are inert |
| `collisionOperator` | 0 or 1 |
| `constraintScheme` | −1 or the one matching the collision operator |
| `xGridScheme`, `xGrid_k` | 5, 0.0 |
| `thetaDerivativeScheme`, `zetaDerivativeScheme` | 2 |
| `NL` (Fokker–Planck) | `min(4, Nxi)` |
| `RosenbluthMethod` | `quadpack` |
| species gradients | gentle enough that the three-surface profile stays positive |

Many SFINCS decks refuse, most often for scheme-1 parameters,
`VMECRadialOption = 1`, full-trajectory switches at finite $E_r$, or
`RHSMode` 2 or 3. Those decks run as they are on the namelist route.

## Transport-matrix decks

For `RHSMode = 2` and `3` the solver loops over `whichRHS`, overwriting the
drives before building each right-hand side exactly as SFINCS v3 does, and all
drives share one multi-right-hand-side solve. For `RHSMode = 3` the speed grid
collapses to the single node $x = 1$ (`Nx = 1`), matching `createGrids.F90`.

```console
dkx sfincs transport-matrix-v3 --input input.namelist --out-matrix L.npy
dkx sfincs monoenergetic-database --input input.namelist --nu-prime 1e-3 1e-2 1e-1 --e-star 0 0.01
```
