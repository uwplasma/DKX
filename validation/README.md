# Validation evidence

This folder holds the recorded evidence behind DKX's validation claims. Nothing
in it is imported at runtime or shipped in the wheel. It is checked in CI by
`tests/test_validation.py` and `tests/test_planning_evidence.py`.

Audit every registered artifact with one command:

```bash
python -m tools.release.registry                       # every entry
python -m tools.release.registry --entry full_kinetic_sfincs
```

That runner checks what is common to every entry (the artifact exists and its
SHA-256 matches the registry, claim scope, schema, exclusions, inputs and
capability agree), then calls the entry's own `audit()`. That audit recomputes
the campaign's gates from the sealed numbers. Most audits also accept
`--results-root <dir>` to re-verify the raw HDF5/NetCDF outputs, which are kept
outside Git and referenced by checksum.

## What is here

| Path | What it is |
| --- | --- |
| `registry.toml` | The index: one entry per artifact with its capability, status, claim, limitations, inputs, command, commits and checksum. |
| `*_v1.json` (20) | Sealed evidence summaries, one per registry entry. Do not edit: the checksum in the registry is the seal. |
| `inputs/` | The exact DKX decks and native cases each artifact was produced from. Artifacts record these paths and checksums, so they cannot move without resealing. |
| `capabilities.toml` | Capability status and open evidence gaps; every registry entry names one. |
| `hardware.toml` | Named measurement hosts; every entry names the host that produced it. |
| `benchmark_schema.toml`, `benchmarks/` | Required fields of a comparable benchmark row, and two measured rows (tier-1 HSX re-measurement; collocation/multigrid h-independence ladder). |
| `baseline.toml` | The Phase A inventory of DKX 2.3.1 (2026-08-30), with the later `[review]` data. |
| `package_size_contract.toml` | Size limits the CI wheel job enforces through `tools/release_contracts.py`. |

The audit code for each entry lives at `tools/paper_benchmarks/audit_*.py` and
is named by the entry's `audit_script`.

## Status vocabulary

- `accepted`: the claim passes its declared gates.
- `accepted_limited`: the claim passes inside an explicitly narrower scope.
- `diagnostic`: the run localizes or bounds a problem without admitting a claim.
- `negative_result`: the run rules a route or resolution out. Kept on purpose
  so that it is not retried.

## Entries

The full claim and limitations of each entry are in `registry.toml`. The
longer write-ups are in `docs/validation_matrix.rst`.

| Entry | Status | Headline |
| --- | --- | --- |
| `independent_cross_code` | accepted_limited | Monoenergetic `D11*`, `D31*`, `D13*`, `D33*` on DSHAPE/NCSX (YANCC) and W7-X EIM (MONKES) within 6%; PAS + DKES only. |
| `full_kinetic_sfincs` | accepted | Full linearized FP, tokamak, `Er = 0`: max scaled DKX/SFINCS error `2.69e-10`, rung movement ≤ `0.280%`. |
| `full_kinetic_sfincs_finite_er` | accepted | Same at normalized `Er = -30`: max error `1.88e-9`, movement ≤ `0.326%`. |
| `full_kinetic_sfincs_stellarator` | accepted | W7-X SC1 at `rN = 0.5`, `Er = 0`: max error `1.37e-8`, movement ≤ `0.444%`. |
| `native_physical_flux_sfincs` | accepted | Corrected native `psiHat → rHat` flux conversion; W7-X at 8.55 kV/m, fluxes within 0.32%, current within 0.024%. |
| `w7x_fixed_field_resolution_referee` | accepted_limited | Fixed-field fluxes admitted at theta15/zeta85/pitch150/speed8 (DKX vs SFINCS pitch-150 within 0.269%); parallel current still theta-sensitive. |
| `native_ambipolar_profile` | accepted_limited | Five-surface W7-X PAS/DKES profile with every root, bracket and recovery retained; not phase-space converged. |
| `w7x_seeded_bracket_discovery` | accepted_limited | Low-resolution discovery gives topology `[1, 3]`; every candidate is replayed to a strict sign change in 8 instead of 98 solves. |
| `w7x_admitted_grid_seeded_envelope` | accepted_limited | Admitted-grid roots 12.681640625 and 11.533203125 kV/m in explicit intervals; not a global root search. |
| `w7x_admitted_grid_uniform_probe_no_go` | negative_result | Uniform all-root search at the admitted grid stopped after 2551 s at 21.9 GB with no surface done: an operational no-go. |
| `ambipolar_phase_space_ladder` | negative_result | Coarse/reference/fine ladder keeps topology, but a root moves 1.626 kV/m and heat flux 7.81%: `refinement_exhausted`. |
| `ambipolar_phase_space_axes` | negative_result | Pitch is the dominant unresolved axis; pitch 44 reached 22.3 GB, so pitch 48 is not admitted. |
| `ambipolar_pitch_budget` | negative_result | Bounded route matches full-factor exactly (≤ `3.6e-11`) at under a tenth of the memory; uniform pitch 22→26→30 changes topology. |
| `ambipolar_pitch_speed_groups` | diagnostic | Three supported pitch-by-speed rules at fixed work give three different root topologies. |
| `ambipolar_pitch_explicit_groups` | diagnostic | Fixing high-speed work holds topology `[1, 3]`, but fields and fluxes still move by up to 1.06 kV/m and 9.9%. |
| `ambipolar_pitch_combined` | negative_result | Raising low and intermediate pitch together keeps `[1, 3]`; further intermediate refinement fails every gate. |
| `ambipolar_speed_local_pitch` | diagnostic | Sensitivity localizes to speed node 3; raising the pitch ceiling exposes high-speed movement. |
| `ambipolar_joint_pitch_speed` | diagnostic | Both the speed 6→8 and pitch 44→52 gates fail; the truncated route reports its modal tail as unavailable. |
| `ambipolar_joint_speed_zeta_tail` | diagnostic | Speed and zeta refinement move fluxes 6.6–9.4%, above the 2% gate; tail bounds are not monotone. |
| `ambipolar_selected_tail_bound` | diagnostic | Rigorous upper bound on the truncated route's Legendre tail at each selected field; supporting evidence only. |

## Adding evidence

Add an entry to `registry.toml` and an `audit()`. Do not add a per-campaign
runner, test module, or README section. Large raw outputs belong in release
assets, referenced by checksum.
