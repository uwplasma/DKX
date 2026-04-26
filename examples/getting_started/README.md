# Getting started

These examples introduce the basic `sfincs_jax` workflow without requiring the Fortran v3 executable.

Suggested order:

1. `build_grids_and_geometry.py` — build v3 grids + geometry objects.
2. `apply_collisionless_operator.py` — apply a collisionless operator slice.
3. `write_sfincs_output_python.py` — write a v3-style `sfincsOutput.h5` from Python.
4. `write_sfincs_output_cli.py` — do the same via the CLI.
5. `write_sfincs_output_tokamak.py` — run the supported analytic tokamak `geometryScheme=1` path.
6. `write_sfincs_output_vmec.py` — run the supported VMEC `geometryScheme=5` path with `wout_path`.
7. `plot_sfincs_output.py` — read an output file and generate a quick summary figure.
8. `write_and_plot_multiple_formats.py` — write `.h5`, `.nc`, and `.npz` outputs and build a PDF diagnostics panel.

Run any script from the repo root, e.g.:

```bash
python examples/getting_started/build_grids_and_geometry.py
```

Common follow-ups:

```bash
python examples/getting_started/write_sfincs_output_tokamak.py
python examples/getting_started/write_sfincs_output_vmec.py
python examples/getting_started/plot_sfincs_output.py
python examples/getting_started/write_and_plot_multiple_formats.py
```
