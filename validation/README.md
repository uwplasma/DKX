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
- `ambipolar_pitch_budget_v1.json`: exact full-versus-bounded uniform-pitch
  route parity plus a bounded pitch-22/26/30 ladder that retains its changing
  topology and `refinement_exhausted` outcome.
- `ambipolar_pitch_speed_groups_v1.json`: fixed-work uniform, linear-ramp,
  and quadratic-ramp pitch allocations on a common W7-X surface pair,
  retaining the topology changes rather than promoting convergence.
- `ambipolar_pitch_explicit_groups_v1.json`: exact-total, exact-high-work
  explicit allocations that isolate low- and intermediate-speed sensitivity
  while retaining the unresolved root and flux movement.
- `ambipolar_pitch_combined_v1.json`: bounded follow-up that raises low and
  intermediate pitch work together, retains the failed intermediate-refinement
  gates, and records the pinned SFINCS/YANCC/MONKES/PENTA source review that
  motivated compact per-speed flux diagnostics.
- `ambipolar_speed_local_pitch_v1.json`: five narrow common-field probes that
  localize the initial change to one intermediate-speed node, then retain the
  failed pitch-36 ceiling and pitch-44 high-speed-tail gates without claiming
  full-profile or phase-space convergence.
- `ambipolar_joint_pitch_speed_v1.json`: speed-6/8 and pitch-44/52 common-field
  probes with route-aware modal-tail evidence. Full states retain the compact
  L2 tail; zero-padded truncated states explicitly mark it unavailable.
- `ambipolar_joint_speed_zeta_tail_v1.json`: a fixed-pitch 2x2 speed/zeta
  matrix with selected-tail upper bounds, exact cold/warm reproducibility, and
  an explicit failed phase-space-convergence outcome.
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

## Bounded uniform-pitch route and ladder

`ambipolar_pitch_budget_v1.json` first checks the memory contract independently
of phase-space convergence. The exact uniform-pitch-22 case switches from 139
full-factor solves to 139 memory-bounded structured solves while retaining
every root and bracket exactly. Selected particle and heat fluxes differ by at
most `3.58e-11` relative, all retained evaluation fluxes by at most `1.53e-10`,
and the bounded maximum true residual is `5.28e-14`. Its cold process takes
`176.33 s`; the retained warm result takes `184.81 s`, so no warm-cache speedup
is claimed. Peak footprint falls from `31,859,925,880 B` to at most
`2,923,810,392 B` in the retained bounded runs.

Audit the compact record, or additionally check all external NetCDF files and
the geometry, with:

```bash
python tools/paper_benchmarks/audit_ambipolar_pitch_budget.py
```

The same bounded route then evaluates uniform pitch 22, 26, and 30 on the
three-surface profile. Root counts change from `[3, 1, 1]` to `[1, 1, 1]` to
`[1, 3, 1]`; adjacent selected fields move by as much as `9.599609375` and
`7.7001953125 kV/m`, and selected heat-flux movements reach `55.72%` and
`45.25%`. Every accepted residual remains below `5.65e-14`, so this is a
discretization failure rather than a solver failure. The artifact keeps the
unchanged `0.005 kV/m`, `2%`, and `1e-12` gates, rejects uniform pitch 34 or
higher, and directs the next diagnostic to isolate speed-node groups at fixed
bounded work. It does not establish phase-space, zeta, speed, independent-code,
full-FP, Phi1, experiment, or performance validation.

## Fixed-work pitch-by-speed diagnosis

`ambipolar_pitch_speed_groups_v1.json` compares only already-supported
allocation rules. Uniform pitch 22, linear-ramp pitch 36, and quadratic-ramp
pitch 44 retain `[22,22,22,22,22,22]`, `[4,9,17,27,36,36]`, and
`[4,5,11,25,44,44]` modes by speed: 132, 129, and 133 active modes in total.
Audit the compact record, or additionally supply its four external NetCDF
files and the pinned Boozer geometry, with:

```bash
python tools/paper_benchmarks/audit_ambipolar_pitch_speed_groups.py
```

On the two common surfaces, root counts change `[3,1] -> [1,3] -> [1,1]`.
Uniform-to-linear holds the intermediate-speed group at exactly 44 modes while
shifting work from low to high speed; selected electric field and heat flux
move by as much as `12.20703125 kV/m` and `68.35%`. Linear-to-quadratic still
changes topology and moves them by `2.177734375 kV/m` and `17.93%`. Every
accepted true residual is below `7.04e-14`, all measured footprints remain
below 4.14 GB, and the quadratic cold/warm scientific arrays are exact apart
from the timing array. The populated-cache process is slower, so no warm
speedup is claimed.

This closes the supported-rule allocation diagnosis, not phase-space
convergence. A subsequent bounded two-surface slice must separate low from
intermediate sensitivity with high-speed work held fixed. Zeta, speed,
independent-code, experiment, full-FP, Phi1, and performance validation remain
open.

## Explicit fixed-high-work diagnosis

`ambipolar_pitch_explicit_groups_v1.json` uses the new deterministic
`resolution.pitch_modes_by_speed` contract to compare the supported linear-36
allocation `[4,9,17,27,36,36]` with low-heavy `[12,12,16,17,36,36]` and
intermediate-heavy `[4,4,24,25,36,36]` allocations. Every allocation has
exactly 129 active modes and exactly 72 modes in the final two speed nodes.
Audit the compact record, or additionally verify all four raw NetCDF files and
the pinned Boozer geometry, with:

```bash
python tools/paper_benchmarks/audit_ambipolar_pitch_explicit_groups.py
```

All three allocations preserve root counts `[1,3]` on the bounded surface
pair, so fixing high-speed work removes the topology change seen between the
supported allocation rules. It does not make the observables converged:
pairwise selected electric-field, particle-flux, and heat-flux movements reach
`1.064453125 kV/m`, `9.89%`, and `9.08%`. All 98 solves in each new result use
the bounded structured route, every accepted residual stays below
`3.05e-14`, and every measured footprint stays below 4.01 GB. The
intermediate-heavy cold/warm scientific arrays are exact apart from timing;
the retained timings support no warm-speedup claim.

The checked outcome remains `refinement_exhausted`. The next bounded pair must
raise low and intermediate work together while retaining the admitted
high-speed group. Zeta, speed, independent-code, experiment, full-FP, Phi1,
and performance validation remain open.

## Joint speed/zeta selected-tail diagnosis

`ambipolar_joint_speed_zeta_tail_v1.json` completes the smallest fixed-pitch
2x2 matrix at speed 8/10 and zeta 37/45. It holds the two W7-X surfaces, three
sampled electric fields, pitch ceiling 52, profiles, physics, and solver policy
fixed. Audit the compact record, or additionally verify all eight cold/warm
NetCDF results and the adjacent pinned geometry, with:

```bash
python tools/paper_benchmarks/audit_ambipolar_joint_speed_zeta_tail.py \
  --results-root ../runtime/evidence/joint-speed-zeta-v1
```

At the common `8.55 kV/m` field, speed refinement changes particle and heat
fluxes by as much as 7.55% and 6.56%; zeta refinement changes them by as much
as 9.32% and 9.41%. Every accepted true residual stays below `1.23e-15`, every
process stays below 2.29 GB RSS, and each cold/warm scientific result is exact
apart from timing. Selected-tail upper bounds remain finite from 7.96% to
9.71%, but decrease on one zeta refinement, so they are not promoted as a
standalone convergence oracle.

The narrow field window observes no bracket by design. This artifact records
`phase_space_converged=false`, admits no full-profile escalation, and directs
the next slice to a matched fixed-field YANCC/SFINCS or MONKES reference before
another DKX grid increase.
