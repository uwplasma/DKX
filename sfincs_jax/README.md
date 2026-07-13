# sfincs_jax Source Layout

This directory contains the importable `sfincs_jax` package. The architecture
is the canonical stack of flat, physics-named root modules
(`plan_final.md`, "Source Structure Rules"): one input file plus one geometry
runs through `inputs -> drift_kinetic -> solve -> moments -> writer/console`,
and the public API/CLI route every supported case through that chain. The
legacy pipeline (the transitional `problems/`, `operators/`, `solvers/`,
`outputs/`, `discretization/`, `geometry/`, and `physics/` packages that were
explicitly transitional interim owners while the vertical slices landed) was
deleted once every SFINCS v3 physics family became canonical: RHSMode 1/2/3,
PAS and full Fokker-Planck collisions, geometry schemes 1-5/11/12/13 (with
lasym), Phi1 (kinetic/collision/readExternalPhi1), constraint schemes -1..4,
export_f, `.npz`/NetCDF output, solver traces, xGridScheme 1-8 with
`xDotDerivativeScheme` -2..11, and magneticDriftScheme 0-9. Only two one-level
packages remain: `validation/` and `workflows/`.

## The Canonical Stack (the architecture)

| Canonical owner | Purpose |
| --- | --- |
| `constants.py`, `species.py` | Normalizations, radial-coordinate Jacobians, species pytrees, collisionality. |
| `phase_space.py`, `xgrid.py` | Theta/zeta grids and derivative matrices, Legendre pitch machinery, speed grids (`xgrid.py` is the polynomial x-grid kernel the collision operators consume), Nxi-for-x ramps. |
| `magnetic_geometry.py` | All supported geometry schemes, VMEC/Boozer readers, differentiable Fourier path. |
| `collisions.py` | Pitch-angle scattering and full Fokker-Planck with Rosenbluth terms. |
| `drift_kinetic.py` | The `KineticOperator`: term assembly, matrix-free apply, analytic Legendre blocks, RHS drives, bordered constraints. |
| `solve.py` | Three-tier policy (structured block elimination, preconditioned recycled Krylov, host direct referee) on the optional `solvax` library; implicit differentiation. |
| `moments.py` | Velocity-space moments, flux families, transport matrices, NTV, classical transport, keyed by sfincsOutput.h5 names. |
| `inputs.py`, `console.py` | Typed namelist with Fortran-cited defaults/validation; byte-parity Fortran stdout blocks. |
| `run.py` | End-to-end RHSMode 1/2/3 drivers (`run_profile`, `run_transport_matrix`, `run_geometry`) plus the namelist-level dispatch `run_from_namelist`. |
| `er.py` | Ambipolar radial-electric-field slice: `radial_current`, Fortran-parity Brent `find_ambipolar_er` (bracket expansion + root classification, warm starts/recycling), and the differentiable `ambipolar_er` (`solvax.implicit.root_solve`). |
| `phi1.py` | Phi1/quasineutrality slice: the nonlinear Newton solve `solve_phi1` (each step linearizes `KineticOperator.residual_phi1` and calls `solve.solve` as the inner linear solve, warm-started), its accepted-iterate history variant `solve_phi1_history` (the writer's per-iteration output), and the differentiable `phi1_state` (`solvax.implicit.root_solve`). Covers `includePhi1InKineticEquation` and `includePhi1InCollisionOperator` (the poloidally varying Fokker-Planck collision operator) with `quasineutralityOption` 1/2. |
| `writer.py` | Canonical `sfincsOutput.h5`/`.nc`/`.npz` writer for RHSMode 1/2/3 (emits `Phi1Hat` for Phi1 runs) and the geometry-only output. |
| `solver_trace.py` | Versioned solver-trace schema with JSON/HDF5 (de)serialization. |
| `variational.py` | Entropy-production variational functionals bounding the monoenergetic transport coefficient from above and below (convergence certificates for the RHSMode=3 output). |
| `shaing_callen.py` | The collisionless (Shaing-Callen / Boozer-Gardner) bootstrap-coefficient limit evaluated from the flux-surface geometry; anchors the low-collisionality physics tests. |
| `bounce_averaged.py` | Differentiable bounce-averaged `1/nu` low-collisionality radial transport (Velasco et al., J. Comput. Phys. 418, 109512 (2020); Nucl. Fusion 61, 116059 (2021)): the effective ripple `epsilon_eff` and the trapped-particle bounce integrals (second adiabatic invariant, bounce-averaged radial magnetic drift) from the `|B|` Boozer spectrum, with spectrally-accurate differentiable bounce points via sine-substitution quadrature (arXiv:2412.01724). Pure JAX, jit/vmap-safe; the `nu -> 0` asymptote of the full-DKE monoenergetic `D11` (a fast optimization surrogate). |
| `monoenergetic.py` | Monoenergetic-database mode: (nuPrime, EStar) scans of the RHSMode=3 coefficients in the benchmark normalization (Beidler et al., Nucl. Fusion 51, 076001 (2011)), the Maxwellian energy convolution to per-species thermal transport matrices, and the compact `.npz` database format (`save_database`/`load_database`); differentiable end to end with the `from_fourier` geometry path. |
| `momentum_correction.py` | Sugama-Nishimura parallel-momentum correction (Phys. Plasmas 9, 4637 (2002); 15, 042502 (2008); Maassberg et al., Phys. Plasmas 16, 072504 (2009)): restores inter-species parallel-momentum conservation lost by the pitch-angle collision operator by coupling the monoenergetic per-species parallel flows through the parallel friction/viscosity matrices and solving a small differentiable linear system for the corrected flows and `<B.j_parallel>`. |
| `api.py`, `cli.py`, `__main__.py` | Thin public surface over the canonical modules. |

The CLI (`write-output` and the bare-run form) dispatches every RHSMode 1/2/3
deck through `run.py`; out-of-range option values are namelist validation
errors raised by `inputs.load_sfincs_input`. There is no legacy fallback.

## Other Root Modules

| Module | Role |
| --- | --- |
| `__init__.py` | Public package exports, JAX precision/cache setup. |
| `ambipolar.py` | Scanplot-compatible ambipolar post-processing (`solve_ambipolar_from_scan_dir`, `radial_current_from_output`) over precomputed scan directories; in-process ambipolar solves live in `er.py`. |
| `batch.py` | First-class batched-solve API over the two canonical batch axes — an `E_r` scan on one geometry (`batched_er_scan`) and a batch of flux surfaces sharing discretization (`batched_surface_scan`) — via `jax.vmap` over the varying `KineticOperator` leaves (`batched_solve`). Reuses `solve.py`/the operator read-only, stays differentiable and jit-safe, and auto-chunks with `jax.lax.map` to a memory-budgeted batch size from the tier-1 footprint model (`solve.tier1_peak_memory_bytes`) and the device/host memory. |
| `sensitivity.py` | JVP/VJP, adjoint, and implicit differentiation helpers. |
| `plotting.py` | Output plotting used by the CLI and examples. |
| `compare.py` | HDF5 comparison, frozen-reference parity, benchmark-table utilities. |
| `io.py`, `namelist.py`, `input_compat.py`, `paths.py` | Output-file reading and generic dict serializers, SFINCS-style namelist parsing, input aliases, data/cache paths. |
| `profiling.py` | Timers and memory probes. |

## Remaining Domain Packages

- `validation/`: frozen-reference loading, Fortran/PETSc fixture readers,
  release-data manifest/fetching, evidence gates, release orchestration.
- `workflows/`: scan-er orchestration (`scans.py`), optimization support
  (`optimization.py`), and the JAX-native geometry adapters for external
  producers (`geometry_adapters.py`).

The deleted legacy stack (the `problems/`, `operators/`, `solvers/`,
`outputs/`, `discretization/`, `geometry/`, and `physics/` packages, the
sparse-direct/CSR-assembly solver families, and the root `grids.py` /
`diagnostics.py` helpers) must not be reintroduced; the canonical `solve.py`
tiers and the flat root modules own the entire supported surface.

## Design Rules

- New stable code goes into a flat, physics-named canonical root module; the
  remaining domain packages hold validation and workflow orchestration only.
- Keep package depth shallow: one folder below `sfincs_jax/`, no nested
  packages.
- Stable file names describe physics or numerics — no version suffixes or
  experiment names (`plan_final.md`, "Source Structure Rules").
- No env-var-only solver routes in stable code; opt-in switches are namelist
  or API arguments with documented semantics.
- Keep large validation data out of the git clone and wheel; large equilibria
  are fetched through `validation.data_fetch` from release assets
  (`validation/equilibria_manifest.json`).

## Stability And Compatibility

The canonical root modules are the stable import surface.
Compatibility aliases may remain in `workflows/__init__.py` only while a
documented workflow needs them.

## Generated Files Policy

Do not commit `__pycache__`, `.pyc`, profiling traces, HDF5/NetCDF/NPZ solve
outputs, XLA profiles, or large equilibrium files inside `sfincs_jax/`.

## Contributor Workflow

1. Start from the slice queue in `../plan_final.md`; delete or extract before
   moving code.
2. Update the source-tree manifest, this file, and the plan in the same
   commit as any module addition, rename, or deletion.
3. Run focused tests plus the import/compile guards before committing; full
   suite at slice milestones.
