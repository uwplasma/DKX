# Features

This page lists what DKX can do, with the scope of each capability and a link
to the evidence behind it. Where a capability is limited, the limit is stated
in the same row.

## Workflows

| Workflow | Entry points | Scope and evidence |
| --- | --- | --- |
| Profile solve (SFINCS `RHSMode = 1`): fluxes, flows, currents per surface and species | native case `run.workflow = "profile"`; `dkx run`; `dkx.run(case)`; `dkx input.namelist` | native SI result; SFINCS parity to 8e-14 on output tables ({doc}`benchmarks/sfincs`); `examples/01_tokamak_profile` |
| Ambipolar profile: $E_r$ from $J_r(E_r) = 0$ on every surface | `run.workflow = "ambipolar_profile"`; `dkx roots`; `dkx.find_ambipolar_er`; `dkx.ambipolar_er` | every root kept and classified; {doc}`tutorials/ambipolar_er`, `examples/05_ambipolar_profile` |
| Thermal transport matrix (`RHSMode = 2`) | `dkx sfincs transport-matrix-v3`; `dkx.run.run_transport_matrix` | 3×3 Onsager matrix; golden-data parity 6e-13 to 9e-9 ({doc}`benchmarks/sfincs`); namelist route only |
| Monoenergetic coefficients (`RHSMode = 3`) and databases over $(\nu', E^*)$ | `dkx sfincs monoenergetic-database`; `dkx.run_monoenergetic_database` | $D_{11}^*$, $D_{31}^*$, $D_{33}^*$ in the Beidler normalization, with variational $D_{11}$ bounds; MONKES/YANCC within 6% ({doc}`benchmarks/cross_code`); `examples/04_monoenergetic_scan` |
| Parameter scans | `[scan]` table and `dkx scan`; `dkx sfincs scan-er`; `dkx.batched_er_scan`, `dkx.batched_surface_scan` | Cartesian or zipped axes, resumable ({doc}`user_guide/scans_and_parallelism`) |
| Convergence study | `dkx converge`; `[convergence]` table | refines each axis and all jointly; {doc}`user_guide/convergence`, `examples/06_convergence_certificate` |
| Adjoint sensitivities (SFINCS `RHSMode = 4/5`) | `dkx.sensitivity` | input validation, output field names and ranks, and compact Fortran replay fixtures; production-grid parity is not claimed |

The native case executor implements the profile and ambipolar workflows. The
`transport_matrix` and `monoenergetic` workflows are in the case format but run
through the SFINCS-deck entry points above.

## Geometry

| Route | How to select it | Notes |
| --- | --- | --- |
| Analytic configurations | `geometry.format = "analytic"`, `file` = `tokamak`, `lhd_standard`, `lhd_inward`, `w7x_standard` | built in; no file needed |
| VMEC `wout` | `format = "vmec"` | stellarator-symmetric and `lasym` equilibria; the LIBSTELL text form is read by `dkx.vmec_ascii`; `examples/02_vmec_stellarator` |
| Boozer `.bc` | `format = "boozer"` | symmetric or asymmetric column layout detected from the file; `examples/03_boozer_stellarator` |
| SFINCS geometry schemes | `geometryScheme` 1–5, 11, 12, 13 in a namelist | {doc}`user_guide/sfincs_namelist` |
| Differentiable Fourier geometry | `FluxSurfaceGeometry.from_fourier` | $\lvert B\rvert$ spectrum as a traced input; `examples/08_vmex_optimization` |

Details in {doc}`physics/geometry` and {doc}`tutorials/stellarator_from_vmec`.

## Physics

| Capability | Options | Evidence |
| --- | --- | --- |
| Collision operators | `pitch_angle_scattering`; `linearized_fokker_planck` (full linearized Fokker–Planck, multispecies, momentum and energy restoring) | conservation to 1e-15, Spitzer–Härm within 0.36% ({doc}`benchmarks/analytic_limits`); {doc}`physics/collisions` |
| Rosenbluth potentials | `quadpack`, `analytic` and `hybrid` routes | agree with each other to 1e-12 on the HSX case of {doc}`benchmarks/sfincs` |
| Trajectory models | `magnetic_drifts = "dkes"` or `"full"`; SFINCS DKES, partial and full trajectories; tangential magnetic drifts (`magneticDriftScheme`) | trajectory sweep reproduced ({doc}`benchmarks/cross_code`) |
| Radial electric field | prescribed, or ambipolar with every root classified ion, electron or unstable; uniform or seeded bracket search | {doc}`physics/electric_field` |
| Species and impurities | any number of species; trace or finite impurities; classical impurity flux (`dkx.classical_impurity_flux`); `dkx.build_impurity_plasma` | impurity flux against Fortran golden data ({doc}`benchmarks/cross_code`); `examples/09_phi1_and_impurities` |
| $\Phi_1$, in-surface potential | quasineutrality options 1 and 2, $\Phi_1$ in the kinetic equation and in the collision operator, Newton–Krylov solve (`dkx.phi1`) | namelist route; native `physics.phi1 = "kinetic"` (Phi1 in the kinetic equation, quasineutrality option 1, `profile` workflow) matches it to 1e-8 (`tests/test_native_phi1.py`); {doc}`physics/phi1_and_impurities` |
| Inductive parallel electric field | `inductiveE` decks | vendored upstream deck `inductiveE_noEr` |
| Distribution function export | SFINCS `export_f` | written by the SFINCS-compatible writer |

## Solvers

| Route | `solve()` method / case `solver.method` | When it applies | Evidence |
| --- | --- | --- | --- |
| Structured direct | `"block_tridiagonal"` / `structured_direct` | operators block-tridiagonal in the Legendre index (pitch-angle scattering, DKES trajectories); truncated kernel keeps $O(K m^2)$ memory | 744,610 unknowns in 27.2 s against 463.6 s for SFINCS ({doc}`benchmarks/performance`) |
| Recycled Krylov | `"gmres"` / `recycled_krylov` | everything else: full Fokker–Planck, tangential drifts, $E_r$ terms, $\Phi_1$; GCROT recycling across solves under a coarse-operator preconditioner | {doc}`numerics/krylov_and_preconditioners` |
| Sparse direct | `"direct"` / `sparse_direct_referee` | any operator up to a few $10^5$ unknowns; exact assembly from operator products, Ruiz equilibration, SuperLU (optional MUMPS) | 66,004 unknowns to 1.3e-14 ({doc}`benchmarks/sfincs`) |
| Factor reuse | `SolveResult.factors`, `solve(..., factors=...)`, `transpose=True` | further right-hand sides and adjoints without refactoring | three solves from zero factorizations ({doc}`numerics/factor_reuse`) |

`method="auto"` picks the route; every solve reports the residual of the
original equation. The routes and their limits are in
{doc}`numerics/solver_routes`.

## Hardware and parallelism

| Capability | How | Evidence |
| --- | --- | --- |
| CPU and GPU | JAX backends; `run.device = "auto"`; float64 throughout | same-host GPU faster at every measured size ({doc}`benchmarks/performance`) |
| Batched solves | `dkx.batched_solve`, `dkx.batched_er_scan`, `dkx.batched_surface_scan`; `parallel.strategy = "batch"` | memory-budgeted chunks from a route-aware footprint estimate |
| Sharding across devices | `parallel.shard = ["surface", "electric_field", "species"]`; `devices=` on batched scans | {doc}`numerics/compilation_and_parallelism` |
| Multi-process | `dkx.initialize_distributed_runtime_from_env`; `--distributed`, `--process-id`, `--process-count` CLI flags | single-case multi-device strong scaling is not claimed |
| Compilation cache | `JAX_COMPILATION_CACHE_DIR` | compile against steady state measured in {doc}`benchmarks/performance` |

## Differentiability and optimization

| Capability | How | Evidence |
| --- | --- | --- |
| Gradients of any solved output | `solve(..., differentiable=True)` under `jax.grad`, `jax.jacfwd`, `jax.jvp`; implicit differentiation with one transposed solve | AD against central differences 4.3e-10, 4.2e-9, 7.9e-11 on three paths ({doc}`tutorials/gradients`) |
| Differentiable ambipolar root | `dkx.ambipolar_er` | implicit function theorem; `tests/test_er.py` |
| Differentiable $E_r$ and profile scans | `dkx.prepare_er_scan(..., differentiable_profiles=True)`, `dkx.batched_er_scan(..., differentiable=True)` | {doc}`tutorials/gradients` |
| Shape derivatives | `FluxSurfaceGeometry.from_fourier` | `examples/08_vmex_optimization` |
| Kinetic bootstrap row in VMEX optimization | `dkx.bootstrap.KineticBootstrapMismatch`, `KineticBootstrapCurrent` | QA objective 1.78 → 0.0061 ({doc}`tutorials/vmex_optimization`) |
| Optimization scripts | `examples/optimization/optimize_*.py` | bootstrap current, electron root, impurity screening |

The VMEC-to-Boozer proxy workflow does not claim full VMEC-boundary-to-kinetic
transport gradients; its scope is machine-checked
({doc}`tutorials/vmex_optimization`). The derivation is in
{doc}`numerics/differentiation`.

## Input, output and tools

| Capability | How |
| --- | --- |
| Native cases | TOML or JSON, deterministic case ID; `dkx template`, `dkx validate` ({doc}`user_guide/inputs`) |
| SFINCS v3 decks | `dkx input.namelist`, `dkx.load_sfincs_input`, `dkx convert` to a native case ({doc}`user_guide/sfincs_namelist`, {doc}`tutorials/sfincs_migration`) |
| Outputs | native NetCDF result in SI; SFINCS-layout HDF5, NetCDF4 or NPZ selected by suffix; `dkx.read_output` ({doc}`user_guide/outputs`) |
| Result certificate | `Result.certificate()`: convergence, route, residual, version, device, precision |
| Comparison | `dkx compare` (NetCDF or HDF5, non-zero exit on difference), `dkx sfincs compare-h5` |
| Plotting and inspection | `dkx plot`, `dkx inspect`, `dkx sfincs plot-output`, upstream `utils/` scripts via `dkx sfincs postprocess-upstream` |
| Environment check | `dkx doctor` |
| Profiling | `DKX_PROFILE=1` phase timings |

The command reference is {doc}`user_guide/cli`.

## SFINCS Fortran v3 feature coverage

How each SFINCS v3 capability maps onto DKX. Statuses: *implemented* (public
entry point plus a focused test or checked artifact), *implemented, with
checks* (production claims limited by stated residual, runtime, memory or
resolution checks), *partial* (spine exists, a Fortran-compatible branch or
production check is incomplete).

| SFINCS v3 capability | Fortran owner | DKX owner | Status |
| --- | --- | --- | --- |
| Namelist schema, defaults, validation | `readInput.F90`, `validateInput.F90` | `dkx.namelist`, `dkx.input_compat` | implemented |
| `RHSMode = 1` profile response | `solver.F90`, `evaluateResidual.F90`, `populateMatrix.F90`, `diagnostics.F90` | `dkx.run` → `dkx.solve` over `dkx.drift_kinetic` | implemented, with checks |
| `RHSMode = 2/3` transport and monoenergetic matrices | `solver.F90`, `diagnostics.F90` | `dkx.run.run_transport_matrix` | implemented, with checks |
| Ambipolar option 2 (Brent) | `ambipolarSolver.F90` | `dkx.er.find_ambipolar_er`, with bracket expansion, warm starts and root classification | implemented |
| Ambipolar options 1 and 3 (Newton with adjoint `dRadialCurrentdEr`) | `ambipolarSolver.F90`, `adjointDiagnostics.F90` | replaced by the differentiable `dkx.er.ambipolar_er` (Brent root, exact implicit derivative) | implemented; Fortran-style replay on small decks only |
| `RHSMode = 4` fixed-$E_r$ sensitivities | `populateAdjointRHS.F90`, `adjointDiagnostics.F90` | `dkx.sensitivity` | partial: derivative spine, JVP/VJP and compact Fortran replay fixtures; production grids are release benchmarks |
| `RHSMode = 5` ambipolar sensitivities | `ambipolarSolver.F90`, `adjointDiagnostics.F90` | `dkx.sensitivity` | partial: compact constant-current fixture; production parity not claimed |
| Collision operators (PAS, full FP) | `populateMatrix.F90` | `dkx.collisions` | implemented |
| Magnetic and electric drift branches | `populateMatrix.F90`, `geometry.F90` | `dkx.drift_kinetic` | implemented, with checks |
| Geometry schemes and radial coordinates | `geometry.F90`, `radialCoordinates.F90` | `dkx.magnetic_geometry` | implemented, with checks |
| $\Phi_1$ and quasineutrality | `evaluateResidual.F90`, `populateMatrix.F90` | `dkx.drift_kinetic`, `dkx.phi1` | implemented; `RHSMode` 4/5 with $\Phi_1$ rejected, as in Fortran |
| Linear solvers | PETSc KSP, MUMPS, SuperLU_DIST | in-house structured direct, sparse direct and recycled Krylov routes; PETSc not required | implemented, with checks |
| Output files | `writeHDF5Output.F90` | `dkx.writer`, `dkx.io`: HDF5, NetCDF, NPZ | implemented |
| Parallelism | MPI through PETSc | JAX CPU/GPU, batching, sharding, multi-process runtime | implemented, with checks; single-case multi-GPU strong scaling deferred |

Upstream suite parity for all of the above is recorded in
{doc}`benchmarks/sfincs`; the status of every validation claim is in
{doc}`benchmarks/validation_matrix`.

## Not supported

- Native-case execution of $\Phi_1$ and of the transport-matrix and
  monoenergetic workflows (use the SFINCS-deck route).
- Production-resolution `RHSMode = 4/5` parity with Fortran.
- Full VMEC-boundary-to-kinetic-transport gradients in the proxy workflow.
- Single-case strong scaling across several GPUs.
- Converging the HSX-like gap deck ($N_x = 16$, 633,604 unknowns) on a 36 GiB
  host; neither DKX nor SFINCS does ({doc}`benchmarks/sfincs`).
