# Transport-matrix workflows (RHSMode=2/3)

These examples demonstrate the **transport-matrix** modes in SFINCS v3:
- `RHSMode=3`: monoenergetic (2×2) transport matrix
- `RHSMode=2`: energy-integrated (3×3) transport matrix

Use the ``dkx postprocess-upstream`` CLI for upstream scanplot workflows
after writing an output file or scan directory.

Examples:
- `transport_matrix_rhsmode2_and_rhsmode3.py` — compute transport matrices and compare to fixtures.
- `transport_matrix_rhsmode2_scheme11_and_scheme5.py` — compare representative scheme-11 and scheme-5 transport-matrix paths.
