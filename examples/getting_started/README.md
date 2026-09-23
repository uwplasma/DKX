# Getting started

These examples introduce the basic `dkx` workflow without requiring the Fortran v3 executable.

Suggested order:

1. `build_grids_and_geometry.py` — build v3 grids + geometry objects.
2. `apply_collisionless_operator.py` — apply a collisionless operator slice.
3. `write_sfincs_output_python.py` — write a v3-style `sfincsOutput.h5` from Python.
4. `write_sfincs_output_cli.py` — do the same via the CLI.
5. `write_sfincs_output_tokamak.py` — run the supported analytic tokamak `geometryScheme=1` path.
6. `write_sfincs_output_vmec.py` — run the supported VMEC `geometryScheme=5` path with `wout_path`.
7. `plot_sfincs_output.py` — read an output file and generate a quick summary figure.

For `.h5`, `.nc`, and `.npz` outputs from one solve, see `../tutorials/run_quick_output_and_plot.py`.

Full solve-and-plot runs (build a deck, solve, read outputs back, and plot):

- `build_input_from_python.py` — build a typed `SfincsInput` programmatically (flat Fortran parameter names), write the namelist, and solve it in memory with `SolverOptions` — no input file needed.
- `run_tokamak.py` — single-species pitch-angle-scattering tokamak, HDF5/NetCDF read-back.
- `run_w7x.py` — two-species Fokker-Planck run on real W7-X Boozer geometry (recycled Krylov route).

Run any script from the repo root, e.g.:

```bash
python examples/getting_started/build_grids_and_geometry.py
```

Common follow-ups:

```bash
python examples/getting_started/write_sfincs_output_tokamak.py
python examples/getting_started/write_sfincs_output_vmec.py
python examples/getting_started/plot_sfincs_output.py
```
