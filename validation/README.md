# DKX 3 planning evidence

These machine-readable records freeze the DKX 2.3.1 starting point for the
DKX 3 roadmap. They are planning and release evidence, not runtime package
data. Large outputs, profiler traces, and external-code builds stay outside
Git; later records refer to them by checksum or stable artifact identifier.

- `capabilities.toml`: initial capability status and evidence gaps.
- `baseline.toml`: repository, package, API, CI, and evidence baseline.
- `hardware.toml`: named measurement hosts and availability.
- `benchmark_schema.toml`: fields required of comparable benchmark rows.
- `independent_cross_code_v1.json`: the accepted, bounded DSHAPE/NCSX/W7-X
  monoenergetic PAS/DKES cross-code rung.
- `native_ambipolar_profile_v1.json`: compact native five-surface W7-X
  PAS/DKES whole-profile workflow certificate.
- `ambipolar_phase_space_ladder_v1.json`: bounded coarse/reference/fine W7-X
  PAS/DKES kinetic-grid ladder that truthfully records exhausted convergence
  gates rather than promoting the reference profile.
- `ambipolar_phase_space_axes_v1.json`: separate theta and pitch rungs that
  diagnose pitch as the dominant unresolved direction and reject a blind
  pitch-48 escalation.
- `inputs/*`: the exact DKX decks and native cases used by these rungs.

## Independent cross-code audit

The first independent device-family rung compares the same zero-field
monoenergetic drift-kinetic equation, Lorentz pitch-angle scattering, and DKES
trajectories in all codes. DSHAPE and NCSX use YANCC; W7-X EIM uses the pinned
MONKES database. The artifact records external commits, geometry/reference
checksums, the applied-collision-frequency mapping, the local-radius Beidler
normalization, handedness conversion, four coefficients, solver residuals,
wall time, and process peak RSS.

Audit every stored number and, when the adjacent YANCC checkout is present,
all external inputs:

```bash
python tools/paper_benchmarks/audit_independent_cross_code_validation.py \
  --yancc-root ../YANCC
```

The 6% gate bounds the recorded cross-discretization spread across ``D11*``,
``D31*``, ``D13*``, and ``D33*`` and may not be relaxed to admit a later
regression. The artifact explicitly excludes full-Fokker-Planck, finite-field,
ambipolar-profile, experimental, and performance-comparison claims. Those
remain separate promotion gates.

## Matched full-kinetic SFINCS rung

`full_kinetic_sfincs_v1.json` closes the next, deliberately narrow gate: a
one-species analytic-tokamak surface with physical density and temperature
gradients, the full linearized Fokker--Planck collision operator, full
trajectories, zero electric field, and SFINCS's recommended automatic
constraint. The `high` and `ultra` decks are exact checked inputs for both DKX
and SFINCS v3.

Audit the compact evidence and DKX-owned input checksums:

```bash
python tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py
```

When the external run tree is available, also verify every raw HDF5 and log:

```bash
python tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py \
  --results-root ../runtime/evidence/full-fp
```

At the finest rung, the largest scaled cross-code error over nonzero scalar and
speed-resolved observables is `2.69e-10`. From `6887` to `12509` unknowns, the
largest retained nonzero-observable movement is `0.280%`; every completed true
residual is below `1.82e-11`. Axisymmetric particle flux and NTV are
cancellation-level quantities, so they are gated by a `1e-12` absolute bound
instead of an unstable relative error. Timings and memory are recorded for
reproducibility but do not support a cross-code performance claim.

This rung does not validate multispecies or stellarator full-Fokker--Planck
physics, finite electric field, Phi1, ambipolar profiles, experiment, or
cross-code performance.

## Matched finite-Er full-kinetic rung

`full_kinetic_sfincs_finite_er_v1.json` applies the same independent workflow
to the pinned upstream one-species full-FP tokamak case at normalized
`Er = -30`. The high and ultra decks use full trajectories, a `1e-13` solver
tolerance, and the exact MUMPS-enabled SFINCS reference build recorded above.
Audit the compact evidence with:

```bash
python tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py \
  --artifact validation/full_kinetic_sfincs_finite_er_v1.json
```

At the finest rung, the maximum scaled difference across flow/current,
momentum flux, heat flux, and the retained speed spectra is `1.88e-9`.
High-to-ultra movement is at most `0.326%`, and every completed true residual
is below `5.25e-11`. Axisymmetric intrinsic ambipolarity leaves the summed
particle flux and NTV at cancellation scale, so they use a `2e-11` absolute
gate. This is one prescribed finite field, not an Er scan or an ambipolar-root
validation, and it does not itself validate stellarator full-FP physics.

## Matched stellarator full-kinetic rung

`full_kinetic_sfincs_stellarator_v1.json` closes the next separate gate on the
checksummed W7-X SC1 Boozer surface at `rN = 0.5`. Both codes use exact
relative-path decks, physical density and temperature gradients, full
linearized Fokker--Planck collisions, full trajectories, zero electric field,
automatic constraint 1, and a `1e-12` solver tolerance. Obtain
`equilibria/w7x-sc1.bc` from the pinned SFINCS commit and verify the SHA-256
recorded in the artifact before running either checked deck.

Audit the compact and deck evidence with:

```bash
python tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py \
  --artifact validation/full_kinetic_sfincs_stellarator_v1.json
```

When the external run tree is available, append
`--results-root ../runtime/evidence/full-fp-stellarator/accepted` to verify the
raw HDF5, cold/warm outputs, and logs. The largest high-to-ultra movement is
`0.444%`; the maximum retained scalar/spectral DKX/SFINCS error is `1.37e-8`,
set by an `8.31e-13` absolute NTV difference, and all completed true residuals
are below `1.82e-12`. Momentum flux is a near-zero absolute gate. Timing and
memory are reproduction metadata, not a cross-code performance claim.

This is a one-species, zero-field surface-profile comparison. It is not an Er
scan, ambipolar root/profile, Phi1, multispecies, experimental, or second
stellarator-family full-FP validation.

## Native whole-profile ambipolar certificate

`native_ambipolar_profile_v1.json` pins the portable physical-unit TOML, W7-X
standard-configuration Boozer checksum, merged DKX commit, compact five-surface
profile, cold/warm native NetCDF checksums, environment, timing, and memory.
Audit the checked compact evidence with:

```bash
python tools/paper_benchmarks/audit_native_ambipolar_profile.py
```

When the staged raw run tree is available, append
`--results-root ../runtime/evidence/native-ambipolar-profile-v1` to verify both
NetCDF files, the external geometry, the compact extraction, and cold/warm
scientific-array identity. The admitted PAS/DKES method case retains all roots,
selected SI fluxes, adaptive evidence, branch events, and both attempts at its
one recovered field. It is not phase-space-converged, continuously localized,
experimental, full-FP, Phi1, independent cross-code ambipolar, or second-family
stellarator validation.

The controlling definitions and acceptance gates are in `../plan.md`.

## Bounded ambipolar phase-space ladder

`ambipolar_phase_space_ladder_v1.json` separates kinetic-grid resolution from
the electric-field midpoint hierarchy. Its three checked physical-unit TOMLs
use `(theta, zeta, pitch, speed)` resolutions `(13, 31, 32, 5)`,
`(15, 37, 36, 6)`, and `(17, 37, 40, 6)` on the same five-surface W7-X
PAS/DKES profile. Audit the compact arithmetic with:

```bash
python tools/paper_benchmarks/audit_ambipolar_phase_space_ladder.py
```

All rungs preserve root counts `[1, 1, 3, 1, 1]`, classifications, branch
identities, and selected branches. The reference-to-fine comparison still
moves one root by `1.6259765625 kV/m`, selected particle flux by `4.08%`, and
selected heat flux by `7.81%`. Those values exceed the unchanged `0.005 kV/m`
and `2%` gates even though every accepted true residual is below `3.92e-13`.
The recorded outcome is therefore `refinement_exhausted`, not phase-space
convergence. The fine rung does not refine zeta or speed beyond the reference,
so it cannot support a hidden full-grid convergence claim.

## Theta/pitch resolution diagnosis

`ambipolar_phase_space_axes_v1.json` retains four exact rungs: the
`(15, 37, 36, 6)` reference, theta-only `(17, 37, 36, 6)`, pitch-only
`(15, 37, 40, 6)`, and the next pitch rung `(15, 37, 44, 6)`. Audit every
root, selected flux, checksum, residual, timing, and memory field with:

```bash
python tools/paper_benchmarks/audit_ambipolar_phase_space_axes.py
```

Theta-only keeps selected particle and heat-flux movement below `2%`, but its
maximum root movement is still `0.1611328125 kV/m`. Pitch-only moves a root by
`1.7333984375 kV/m` and selected heat flux by `9.47%`. Pitch 40 to 44 remains
far outside the gates: `0.205078125 kV/m`, `13.52%` selected particle flux,
and `14.07%` selected heat flux. The pitch-44 process reached a
`22,275,409,800 B` footprint, so a brute-force pitch-48 run is not admitted.
The status remains `refinement_exhausted`; zeta and speed remain untested.
