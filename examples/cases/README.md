# Native Case format

The folder contains schema-v1 cases with physical field names and explicit
engineering units. The analytic prescribed-field, adaptively refined analytic
ambipolar, VMEC prescribed-field, and Boozer prescribed-field profiles are
directly executable. The higher-resolution W7-X case demonstrates the later
full-drift and sharding contract.

The ambipolar example retains all roots and adaptive evaluations. Its native
Result also assigns radial branch IDs, records selection reasons and any
creation/loss/merger/crossing/classification events, and flags nonsmooth event
intervals without hiding alternative roots.

```console
dkx validate examples/cases/w7x_ambipolar_profile.toml
python - <<'PY'
import dkx

result = dkx.run(dkx.Case.from_file("examples/cases/vmec_profile.toml"))
result.save()
PY

dkx run examples/cases/boozer_profile.toml
dkx run examples/cases/analytic_ambipolar_profile.toml
```

All executable cases bypass the SFINCS namelist adapter. The VMEC case
reads its `wout` once. The Boozer case reads and parses its `.bc` once,
auto-detects the checked cosine-only or asymmetric column convention, and does
not expose a SFINCS geometry-scheme number in the case. Both reuse one
shape-stable phase-space grid across their surfaces. Replace the checked fixture
paths with a VMEC `wout_*.nc` or Boozer `.bc`; relative paths are resolved beside
the TOML file, not from the shell's current directory.

Generate the same commented format with `dkx schema --format toml`.
