# Native Case format

[`w7x_ambipolar_profile.toml`](w7x_ambipolar_profile.toml) is the complete
schema-v1 profile from the DKX 3 plan. It uses physical field names and explicit
engineering units, and it can be checked before the illustrative geometry is
downloaded:

```console
dkx validate examples/native/w7x_ambipolar_profile.toml
```

Generate the same commented format with `dkx schema --format toml`. This folder
currently demonstrates the native input contract only; native numerical
dispatch and `Result` output land in the next vertical slice.
