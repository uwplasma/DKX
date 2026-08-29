# Native Case format

The folder contains schema-v1 cases with physical field names and explicit
engineering units. The analytic prescribed-field, analytic ambipolar, and VMEC
prescribed-field profiles are directly executable. The higher-resolution W7-X
case demonstrates the later full-drift, convergence, and sharding contract.

```console
dkx validate examples/native/w7x_ambipolar_profile.toml
python - <<'PY'
import dkx

result = dkx.run(dkx.Case.from_file("examples/native/vmec_profile.toml"))
result.save()
PY

dkx run examples/native/analytic_ambipolar_profile.toml
```

All executable native cases bypass the SFINCS namelist adapter. The VMEC case reads
its `wout` once and reuses one shape-stable phase-space grid across all three
surfaces. Replace its checked fixture path with any VMEC `wout_*.nc`; relative
paths are resolved beside the TOML file, not from the shell's current directory.

Generate the same commented format with `dkx schema --format toml`.
