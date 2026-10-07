# Outputs

The two input routes produce two kinds of output.

| route | in memory | on disk |
|---|---|---|
| native case | `dkx.Result` | NetCDF4, DKX result schema 1 (`Result.save`, `dkx run`) |
| SFINCS namelist | `ProfileRun`, `TransportRun`, `MonoenergeticDatabase` | `sfincsOutput.h5` (or `.nc`, `.npz`); `.npy` and `.npz` from the transport commands |

## The `Result` object

`dkx.run(case)` returns an immutable `dkx.Result` (`src/dkx/result.py`):

| attribute | content |
|---|---|
| `case_id` | SHA-256 of the canonical case ({doc}`inputs`) |
| `case_name`, `workflow` | from the case |
| `arrays` | read-only mapping of name to NumPy array |
| `dimensions` | mapping of array name to its tuple of named axes |
| `metadata` | provenance and solver evidence (below) |
| `warnings` | tuple of strings; empty when nothing needs flagging |
| `output_path` | where `save()` writes by default, or where `load()` read from |
| `schema_version` | `1` |

Arrays are reached by key or attribute: `result["heat_flux_W_m2"]` and
`result.heat_flux_W_m2` are the same array. Array names carry their unit.
There is no xarray dependency; `dimensions` names every axis.

| method | effect |
|---|---|
| `print_summary()` | plain-text summary: name, ID prefix, workflow, convergence, residual, route, total time, ambipolar root and refinement counts, array names, warnings |
| `certificate()` | compact dict for review: case ID, the provenance metadata keys, `primal_residual`, `primal_rhs_norm`, warnings |
| `save(path=None, overwrite=True)` | write NetCDF4 (`.nc` suffix required) |
| `Result.load(path)` | read a schema-1 file back |
| `plot(path=None)` | particle flux, heat flux, $\langle j_\parallel B\rangle$ and $E_r$ against $\psi_N$, with every ambipolar branch overlaid; returns the figure, or writes it when a path is given |
| `to_dict()` | metadata, dimensions and array shapes/dtypes (no data) |
| `operator` | the last solved kinetic operator; only on a result from `dkx.run`, not on a loaded one |

### Profile arrays

Every native result has these:

| array | dimensions | unit | meaning |
|---|---|---|---|
| `surface` | surface | 1 | normalized toroidal flux $\psi_N$ |
| `r_N` | surface | 1 | $r/a = \sqrt{\psi_N}$ |
| `species` | species | | species names |
| `charge_e` | species | $e$ | charges |
| `mass_amu` | species | u | masses |
| `density_m3` | surface, species | m$^{-3}$ | input density profile |
| `temperature_keV` | surface, species | keV | input temperature profile |
| `electric_field_kV_m` | surface | kV/m | prescribed $E_r$, or the selected ambipolar root |
| `particle_flux_m2_s` | surface, species | m$^{-2}$ s$^{-1}$ | radial particle flux density $\langle\boldsymbol\Gamma_s\cdot\nabla r\rangle$ |
| `heat_flux_W_m2` | surface, species | W m$^{-2}$ | radial heat flux density $\langle\mathbf Q_s\cdot\nabla r\rangle$ |
| `parallel_current_A_T_m2` | surface | A T m$^{-2}$ | $\langle j_\parallel B\rangle$, the VMEC `jdotb` unit (bootstrap current) |
| `primal_residual` | surface | | $\lVert Ax-b\rVert_2$ of the accepted state, recomputed from the original operator |
| `primal_rhs_norm` | surface | | $\lVert b\rVert_2$ |
| `solver_iterations` | surface | | Krylov iterations (0 for a direct route); for ambipolar runs, the number of kinetic evaluations |
| `solve_time_s` | surface | s | wall time of the solves on that surface |

The fluxes are the SFINCS `particleFlux_vm_psiHat`, `heatFlux_vm_psiHat` and
`FSABjHat` converted with the factors in [the SI table](#converting-hat-quantities-to-si).
The relative residual of each surface is `primal_residual / primal_rhs_norm`;
a result is returned only when it is at most `solver.relative_tolerance`.

### Ambipolar arrays

An `ambipolar_profile` result keeps every kinetic evaluation, every bracket and
every root, not only the selected one. The extra dimensions are `evaluation`,
`root`, `refinement`, `solver_attempt`, `branch_event`,
`branch_event_participant`, `bracket_endpoint` (2) and `speed`. Arrays are
padded along these dimensions (NaN, empty string, or −1 for integer levels) where a surface has fewer
entries.

| array | dimensions | unit | meaning |
|---|---|---|---|
| `ambipolar_status` | surface | | `bracketed_root`, `no_bracketed_root`, or a `seeded_bracket_*` status |
| `ambipolar_root_count` | surface | | roots found on the surface |
| `ambipolar_root_kV_m` | surface, root | kV/m | root positions |
| `ambipolar_root_type` | surface, root | | `ion` ($E_r<0$), `electron` ($E_r>0$) or `unstable` (negative slope) |
| `ambipolar_root_slope_A_m2_per_kV_m` | surface, root | A m$^{-2}$ per kV/m | $dJ_r/dE_r$ at the root |
| `ambipolar_root_current_A_m2` | surface, root | A m$^{-2}$ | residual radial current at the accepted root |
| `ambipolar_root_bracket_kV_m` | surface, root, bracket_endpoint | kV/m | final bracket |
| `ambipolar_root_final_bracket_width_kV_m` | surface, root | kV/m | its width |
| `ambipolar_root_branch_id` | surface, root | | stable branch label across surfaces |
| `selected_ambipolar_root`, `selected_ambipolar_branch` | surface | | which root fills `electric_field_kV_m` |
| `ambipolar_selection_reason` | surface | | why that root was selected |
| `evaluation_electric_field_kV_m` | surface, evaluation | kV/m | every $E_r$ at which the kinetic equation was solved |
| `radial_current_A_m2` | surface, evaluation | A m$^{-2}$ | $J_r = \sum_s Z_s e\,\Gamma_s$ at each evaluation |
| `evaluation_particle_flux_m2_s`, `evaluation_heat_flux_W_m2` | surface, evaluation, species | as above | fluxes at each evaluation |
| `evaluation_parallel_current_A_T_m2` | surface, evaluation | A T m$^{-2}$ | current at each evaluation |
| `evaluation_particle_flux_m2_s_vs_speed`, `evaluation_heat_flux_W_m2_vs_speed` | surface, evaluation, speed, species | as above | speed-node contributions; summing over `speed` gives the flux |
| `speed_v_th` | speed | 1 | speed nodes $x = v/v_{th}$ |
| `evaluation_reason`, `evaluation_stage`, `evaluation_refinement_level` | surface, evaluation | | why each evaluation was made |
| `evaluation_primal_residual` | surface, evaluation | | original-equation residual of each evaluation |
| `evaluation_solver_attempt_*` | surface, evaluation, solver_attempt | | requested and executed method, residual, acceptance, reason of each attempt |
| `ambipolar_refinement_status` | surface | | `resolved`, `refinement_exhausted`, `no_bracket_observed` or `not_requested` |
| `ambipolar_refinement_*` | surface, refinement | | per-level root count, root movement, observable movement, bracket width, evaluation counts |
| `ambipolar_branch_event_*` | surface, branch_event (and participant) | | `boundary_origin`, `creation`, `loss`, `merger`, `crossing`, `classification_transition`, with the branches, field and a nonsmooth flag |
| `ambipolar_nonsmooth_event` | surface | | a branch event makes the profile non-differentiable here |
| `ambipolar_search_scope`, `ambipolar_search_strategy` | surface | | global grid or `explicit_seeded_intervals_only` |

`no_bracketed_root` means the finite search saw no sign change of $J_r$; the
sampled point with the smallest $|J_r|$ is kept in `electric_field_kV_m` and is
never called ambipolar. Sign sampling cannot see a tangential root or an even
number of crossings between two samples, so it is not proof that no root
exists. `ambipolar_refinement_status = "resolved"` requires the final two
refinement levels to keep the same nonzero root count within the declared
root, observable and bracket-width tolerances.

`dkx roots RESULT` prints the root table and branch events;
`dkx plot RESULT --kind search` draws $J_r$ against $E_r$ with every
evaluation, bracket and root.

### Metadata

| key | content |
|---|---|
| `canonical_case` | the normalized case (`Case.to_dict()`) |
| `converged` | `true` when every surface met the residual test |
| `solver_route` | the route that ran, or a list when surfaces differ (for example `block_tridiagonal_truncated`) |
| `route_reason` | how the route was chosen |
| `residual_norm` | largest `primal_residual` (or `evaluation_primal_residual`) |
| `original_residual_evidence` | names of the residual and RHS-norm arrays, the norm (`absolute_l2`), the tolerance, whether the state was complete |
| `iterations` | sum of `solver_iterations` |
| `normalization` | reference density ($10^{20}$ m$^{-3}$), temperature (1 keV), mass (kg), `a_hat`, `psi_a_hat` |
| `phase_space` | pitch allocation source, `pitch_speed_ramp`, active pitch modes per speed node and their sum |
| `geometry_sha256` | hash of the VMEC/Boozer file (of the model name for analytic geometry) |
| `dkx_version`, `python_version`, `jax_version`, `jaxlib_version`, `platform` | software provenance |
| `precision`, `device` | `float64`, and `platform:device_kind` of the device used |
| `timings_s` | `solve` and `total` seconds |
| `peak_host_memory_bytes` | peak resident memory of the process |
| `ambipolar_*` | search, selection, branch-continuation, refinement and solver-attempt summaries (`null` for a profile run) |
| `legendre_tail_diagnostic` | which Legendre-tail diagnostic was retained |

`certificate()` returns the subset a reviewer needs to reproduce and audit the
run. It records algebraic acceptance and provenance; it does not establish
grid convergence or the validity of a root branch. Run `dkx converge` for the
former ({doc}`convergence`).

### Scan results

`dkx scan` writes one `Result` with workflow `scan:<workflow>` and a single
dimension `case`:

| array | meaning |
|---|---|
| `case_id` | ID of each derived case |
| `status` | `ok`, or `failed: <exception>` |
| `axis_<path>` | the value of each scan axis (`.` and `[` become `_`) |
| `particle_flux_m2_s`, `heat_flux_W_m2`, `parallel_current_A_T_m2` | the **largest absolute value** over surfaces and species of each derived run |

The scan arrays are summaries for plotting and resumption, not the full
profiles; run a single case to obtain its arrays. Metadata holds
`scan_combine`, `scan_axes`, `scan_cases`, `scan_succeeded` and `scan_failed`.

## The NetCDF file

`Result.save` writes a NetCDF4 file:

- global attributes `dkx_result_schema` (1), `case_id`, `case_name`,
  `workflow`, `metadata_json` (the metadata as JSON) and `warnings_json`;
- one NetCDF dimension per named axis;
- one variable per array on its named dimensions; numeric arrays with at
  least one dimension are zlib-compressed and string arrays are variable-length
  strings.

The file opens with any NetCDF reader:

```python
import json, netCDF4
with netCDF4.Dataset("result.nc") as f:
    flux = f["particle_flux_m2_s"][:]          # (surface, species)
    meta = json.loads(f.getncattr("metadata_json"))
```

`Result.load` refuses a file whose `dkx_result_schema` it does not know rather
than guessing.

## SFINCS output files

The namelist route writes the SFINCS v3 output layout, so upstream
post-processing reads it. The file suffix picks the format:

| suffix | format |
|---|---|
| `.h5`, `.hdf5` | HDF5 with the Fortran array layout (the default, and the parity format) |
| `.nc`, `.netcdf` | NetCDF4 with the same layout; the deck is the `input_namelist` global attribute |
| `.npz` | uncompressed NumPy archive |

```console
dkx input.namelist --out sfincsOutput.h5
dkx sfincs write-output --input input.namelist --out sfincsOutput.nc
```

```python
out = dkx.write_output("input.namelist", "sfincsOutput.h5")
data = dkx.read_output(out)          # dict for .h5, .nc or .npz
```

Fortran writes column-major arrays, so a $(\theta,\zeta)$ array reads back in
`h5py` as $(\zeta, \theta)$. DKX writes the same layout by default so files
compare directly with Fortran output; `--no-fortran-layout` disables it.

In the shapes below S is the number of species, T is `Ntheta`, Z is `Nzeta`,
X is `Nx` and N is the number of right-hand sides (1 for `RHSMode = 1`).

| group | variables |
|---|---|
| grids | `theta` (T), `zeta` (Z), `x` (X), `Nxi_for_x` (X) |
| geometry scalars | `NPeriods`, `B0OverBBar`, `GHat`, `IHat`, `iota`, `VPrimeHat`, `FSABHat2`, `diotadpsiHat` |
| geometry fields (Z,T) | `BHat`, `DHat`, `dBHatdtheta`, `dBHatdzeta`, `dBHatdpsiHat`, `BHat_sub_*`, `BHat_sup_*`, `BDotCurlB`, `uHat`, `gpsiHatpsiHat` |
| normalization and surface | `Delta`, `alpha`, `nu_n`, `EParallelHat`, `Er`, `dPhiHatd{psiHat,psiN,rHat,rN}`, `psiHat`, `psiN`, `rHat`, `rN`, `psiAHat`, `aHat` |
| species (S) | `Zs`, `mHats`, `THats`, `nHats`, `dnHatd*`, `dTHatd*`; `nuPrime`, `EStar` for `RHSMode = 3` |
| radial fluxes (S,N) | `particleFlux_vm_psiHat`, `heatFlux_vm_psiHat`, `momentumFlux_vm_psiHat`, their `_vm0` variants, and each in `_psiN`, `_rHat`, `_rN`; `*_vs_x` (X,S,N) |
| flows and current | `FSABFlow` (S,N), `FSABjHat` (N), `FSABjHatOverB0`, `FSABjHatOverRootFSAB2`, `flow`, `jHat`, `densityPerturbation`, `pressurePerturbation` (Z,T,S,N) |
| transport matrix | `transportMatrix` ($3\times3$ for `RHSMode = 2`, $2\times2$ for `RHSMode = 3`), stored transposed |
| $\Phi_1$ | `Phi1Hat` (Z,T,1) with `includePhi1` |
| other | `classicalParticleFlux*`, `classicalHeatFlux*` (VMEC and Boozer geometry), `NTV`, `sources`, `elapsed time (s)`, the `input.namelist` text |
| solver record | `linearSolverMethod`, `linearSolverRequestedMethod`, `linearSolverResidualNorm`, `linearSolverResidualTarget` |

The flux suffix `_vm` is the magnetic ($\nabla B$ and curvature) drift acting on
the whole distribution, `_vm0` the same acting on the leading-order part only.
A `_psiHat` flux becomes `_psiN`, `_rHat` or `_rN` by the factor
$d(\text{coordinate})/d\hat\psi$. There is no `radialCurrent` variable:
ambipolarity is $\sum_s Z_s\Gamma_s = 0$ on the per-species particle fluxes.
The $E\times B$ flux diagnostics `_vE` are not written.

### Converting hat quantities to SI

With the reference set of {doc}`sfincs_namelist`:

| output | multiply by | gives |
|---|---|---|
| `FSABjHatOverRootFSAB2` | `dkx.units.CURRENT_DENSITY` $= e\bar n\bar v$ | $\langle j_\parallel B\rangle/\sqrt{\langle B^2\rangle}$ [A m$^{-2}$] |
| `FSABjHat` | `dkx.units.PARALLEL_CURRENT` $= e\bar n\bar v\bar B$ | $\langle j_\parallel B\rangle$ [A T m$^{-2}$] |
| `particleFlux_*_rHat` | `dkx.units.PARTICLE_FLUX` $= \bar n\bar v$ | $\langle\boldsymbol\Gamma\cdot\nabla r\rangle$ [m$^{-2}$ s$^{-1}$] |
| `heatFlux_*_rHat` | `dkx.units.HEAT_FLUX` $= 2\bar n\bar T\bar v$ | $\langle\mathbf Q\cdot\nabla r\rangle$ [W m$^{-2}$] |
| `nHats`, `THats`, `mHats` | $10^{20}$, 1, $m_p/\mathrm{u}$ | m$^{-3}$, keV, u |
| `Er` | 1 | kV/m |

The factors follow from the v3 technical documentation (eqs. 98, 194–196, 201,
220–221; `src/dkx/units.py`). Take the `_rHat` flux, or convert a `_psiHat`
one with `dkx.units.flux_psi_hat_to_r_hat(psi_a_hat=..., a_hat=..., r_n=...)`,
the `ddrHat2ddpsiHat` factor of `diagnostics.F90`. The $\bar R$ in the flux
factor and in $\hat r = r/\bar R$ cancel, so the SI flux density is exactly
the factor times the `_rHat` value. The native `Result` applies these same
conversions.

### Transport-matrix and monoenergetic outputs

| command or call | output |
|---|---|
| `dkx sfincs transport-matrix-v3 --input D --out-matrix L.npy` | the transport matrix as `.npy`; `--out` also writes a SFINCS output file, `--out-state-prefix` the solution vectors |
| `dkx.run_transport_matrix(D)` | `TransportRun`: `transport_matrix`, `state_vectors` (N, unknowns), `moments`, `solve_result`, `operator` |
| `dkx sfincs monoenergetic-database --input D --nu-prime ... --e-star ...` | `.npz` database |
| `dkx.run_monoenergetic_database(D, nu_prime_grid, e_star_grid)` | `MonoenergeticDatabase` |

The monoenergetic database holds the grids `nu_prime` ($n_\nu$) and `e_star`
($n_E$), the star-normalized coefficients `d11_star`, `d13_star`, `d31_star`,
`d33_star` (each $n_\nu\times n_E$), `nu_star`, `v_e`, the geometry scalars as
`metadata_json`, and the deck text. `examples/tutorials/09_monoenergetic.py` plots
$D_{11}^*$, $D_{31}^*$ and $D_{33}^*$ against $\nu'$.

## Inspecting and comparing results

| command | reads | does |
|---|---|---|
| `dkx inspect RESULT` | DKX `.nc` | lists workflow, schema, convergence and every array with its shape and dtype |
| `dkx plot RESULT` | DKX `.nc` or SFINCS output | writes a PNG beside the input (`--out` to choose); `--kind summary`, `search` or `scan` for a DKX result |
| `dkx roots RESULT` | DKX `.nc` | ambipolar roots, classification, bracket width, branch; flags nonsmooth branch events |
| `dkx compare A B` | two DKX `.nc` or two SFINCS `.h5` | compares every array within `--rtol` (default `1e-9`) and `--atol` (default 0); exits non-zero on any difference |
| `dkx --plot sfincsOutput.h5` | SFINCS output | multi-page PDF diagnostics panel |

`dkx compare` refuses a DKX result against a SFINCS file, because their
variable names differ and an empty match would read as agreement. Wall time
and iteration counts are reported as informational and never decide the
verdict. Details of every command are in {doc}`cli`.
