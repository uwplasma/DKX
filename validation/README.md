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

The controlling definitions and acceptance gates are in `../plan.md`.
