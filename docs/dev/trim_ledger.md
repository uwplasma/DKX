# Source-trim ledger (reachability audit, main)

Package 100 files / ~101k lines. Canonical compute path (run+er+phi1 closure) =
19 modules / 13.5k lines, ISOLATED from problems/operators/solvers. It pulls
only 3 live files out of legacy — to be MOVED into flat modules, not deleted:
physics/collisions.py (1137), discretization/xgrid.py (272),
validation/data_fetch.py (194).

Legacy stays reachable through exactly ONE door (2026-07-11): the
`io.write_sfincs_jax_output_h5` write fallback (`cli solve-v3` now routes
canonical). After the 2026-07-11 slice series (Phi1 collision coupling,
export_f/.npz/solver-trace, vE/vd writer families, geometryScheme 13,
constraintScheme 3/4, readExternalPhi1, non-stell-sym VMEC) the fallback
triggers ONLY on: magneticDriftScheme 2-9, xGridScheme outside {1,2,5,6},
`xDotDerivativeScheme != 0`, and three legacy-only CLI options
(`--geometry-only`, `--no-fortran-layout`, `--no-overwrite`). Decision:
implement the remaining two physics features, then delete the gated bulk.

## Near-term (no physics gate)
- validation/suite.py(4199): retire to the reference-data-v2 gate — CLEAN (0
  production importers, top-of-chain; its 3 tests exercise suite tooling, not
  physics parity). CORRECTION: artifacts.py(3346) and release.py(2329) are NOT
  retirable near-term — import-reachability missed that artifacts.py is a live
  examples/publication_figures dependency (hard-required by REQUIRED_CORE_SLIM
  owners), and release.py hosts 6 live `python -m ...release` CLI tools
  (repo-size hygiene gate, production-input generation, write-output-trace,
  readme-audit) imported by 9 mostly-non-slow tests. release.py can only be
  trimmed after its live tools are relocated to durable homes (a structural
  slice); artifacts.py retirement is an examples/-touching task.
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
