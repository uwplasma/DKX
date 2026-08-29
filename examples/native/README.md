# Native Case format

The folder contains three schema-v1 cases with physical field names and
explicit engineering units. The analytic and VMEC prescribed-field profiles
are directly executable; the W7-X ambipolar profile demonstrates fields whose
native solver is still planned.

```console
dkx validate examples/native/w7x_ambipolar_profile.toml
python - <<'PY'
import dkx

result = dkx.run(dkx.Case.from_file("examples/native/vmec_profile.toml"))
result.save()
PY
```

Both executable cases bypass the SFINCS namelist adapter. The VMEC case reads
its `wout` once and reuses one shape-stable phase-space grid across all three
surfaces. Replace its checked fixture path with any VMEC `wout_*.nc`; relative
paths are resolved beside the TOML file, not from the shell's current directory.

Generate the same commented format with `dkx schema --format toml`.
