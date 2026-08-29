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
- `inputs/*.namelist`: the exact DKX decks used by that rung.

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

The controlling definitions and acceptance gates are in `../plan.md`.
