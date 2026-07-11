# Source-trim ledger (reachability audit, main)

Package 100 files / ~101k lines. Canonical compute path (run+er+phi1 closure) =
19 modules / 13.5k lines, ISOLATED from problems/operators/solvers. It pulls
only 3 live files out of legacy — to be MOVED into flat modules, not deleted:
physics/collisions.py (1137), discretization/xgrid.py (272),
validation/data_fetch.py (194).

Legacy stays reachable through exactly TWO doors:
- `cli solve-v3` (direct legacy RHSMode-1 entry) — CLOSE by routing to run.run_profile
- `io.write_sfincs_jax_output_h5` (write fallback for deferred decks)

## Near-term (~13,800 lines, no physics gate)
- validation/suite.py(4199)+artifacts.py(3346)+release.py(2329): retire to the
  reference-data-v2 gate (0 production importers) — ~8000, med risk (41 test files)
- profile_policies.py: 281 SFINCS_JAX_* research toggles — trim ~2000
- workflows/optimization.py(2021)+geometry/jax_adapters.py(964): relocate to
  examples/ — 2985 (AFTER the optimization-examples agent finishes)
- root collisions.py (791): DEAD (0 production importers) — delete + migrate test;
  it duplicates the LIVE physics/collisions.py which should take the flat slot

## Gated bulk (~68k, delete-when-feature-canonical + tests migrate)
- problems/ (32,079): OLD RHSMode 1/2/3 default, superseded by run.py+solve.py
- solvers/ (16,742): krylov + 12 preconditioner_* — DEFER TO SOLVAX (solve.py
  already uses solvax gcrot/block-Thomas/splu/schur)
- operators/ (7,689): legacy V3 operator assembly, superseded by drift_kinetic.py
- outputs/ legacy writer (5,844): superseded by canonical writer.py
- geometry/ (3,053): FULLY duplicated by magnetic_geometry.py
- discretization/v3 etc (1,393), mapped-xgrid (994), classical_transport (126)

## Duplication flags (legacy that should already be gone)
1. root collisions.py (dead) vs physics/collisions.py (live) — flat slot holds dead gen
2. magnetic_geometry.py fully duplicates the geometry/ package
3. root writer.py vs outputs/ writer stack
4. inputs/phase_space/magnetic_geometry vs discretization/v3 grids/geometry-from-namelist

Combined removable ceiling ~81,700 lines -> ~30k canonical end-state.
