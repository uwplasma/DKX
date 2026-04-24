# Publication-style figures

These scripts generate polished figures (PNG + PDF) intended to be *paper-ready*.

Requirements:

```bash
pip install matplotlib
```

Outputs are written under `examples/publication_figures/figures/` by default.
The literature-facing validation/figure map is tracked in
`examples/publication_figures/validation_manifest.json`.

Examples:
- `magnetic_drifts_publication_figures.py`
- `er_terms_publication_figures.py`
- `generate_sfincs_paper_figs.py`
- `generate_er_trajectory_sweep.py`
- `generate_w7x_ambipolar_validation.py`
- `generate_autodiff_sensitivity_validation.py`

Pinned fixed-case runs on the refactor branch:

```bash
python examples/publication_figures/generate_er_trajectory_sweep.py \
  --preset tokamak_like \
  --er-values=-30,0,30 \
  --work-dir examples/publication_figures/output/er_sweep_tokamak_reference \
  --summary-json examples/publication_figures/output/er_sweep_tokamak_reference/summary.json \
  --out-dir docs/_static/figures/paper \
  --stem sfincs_jax_er_trajectory_sweep_tokamak_reference

python examples/publication_figures/generate_er_trajectory_sweep.py \
  --preset stellarator_like \
  --fast \
  --er-values=-8.5897,0,8.5897 \
  --work-dir examples/publication_figures/output/er_sweep_stellarator_fast_reference \
  --summary-json examples/publication_figures/output/er_sweep_stellarator_fast_reference/summary.json \
  --out-dir docs/_static/figures/paper \
  --stem sfincs_jax_er_trajectory_sweep_stellarator_fast_reference
```

Those pinned runs are checked in as:
- `examples/publication_figures/artifacts/er_sweep_tokamak_reference_summary.json`
- `examples/publication_figures/artifacts/er_sweep_stellarator_fast_reference_summary.json`
- `docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep_tokamak_reference.png`
- `docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep_stellarator_fast_reference.png`

Corrected bounded collisionality branch artifact:

```bash
python examples/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --fast \
  --scan-only \
  --work-dir examples/publication_figures/output/lhd_reaudit_fast
```

`generate_sfincs_paper_figs.py` now writes machine-readable collisionality summaries
to `--summary-dir` as well. If `--summary-dir` is omitted, the summaries are written
into the selected `--work-dir` with top-level `metadata` and sorted `rows`.

For the heavy full-resolution re-audit lanes, the generator now supports
split-operator execution so the FP and PAS ladders can be resumed independently
on separate devices:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false SFINCS_JAX_SCAN_RECYCLE=1 \
python examples/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --collision-operators 0 \
  --skip-existing \
  --scan-only \
  --work-dir examples/publication_figures/output/lhd_reaudit_full \
  --summary-dir examples/publication_figures/artifacts

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false SFINCS_JAX_SCAN_RECYCLE=1 \
python examples/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --collision-operators 1 \
  --skip-existing \
  --scan-only \
  --work-dir examples/publication_figures/output/lhd_reaudit_full \
  --summary-dir examples/publication_figures/artifacts
```

Once both operator ladders have completed, synthesize the audited summary and
figures without rerunning the scan:

```bash
python examples/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --plot-only \
  --collision-operators 0,1 \
  --work-dir examples/publication_figures/output/lhd_reaudit_full \
  --summary-dir examples/publication_figures/artifacts \
  --out-dir docs/_static/figures/paper
```

The same split-and-synthesize pattern applies to the W7-X collisionality lane.
When ``--skip-existing`` is used, the generator keeps completed scan points and
prunes only stale subdirectories that do not contain ``sfincsOutput.h5`` before
rerunning the missing points.

The corrected bounded LHD rerun is currently pinned as:
- `examples/publication_figures/artifacts/lhd_collisionality_reaudit_fast_summary.json`
- `docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality_reaudit_fast.png`

The corrected bounded W7-X rerun is currently pinned as:
- `examples/publication_figures/artifacts/w7x_collisionality_reaudit_fast_summary.json`
- `docs/_static/figures/paper/sfincs_jax_fig2_w7x_collisionality_reaudit_fast.png`

The full collisionality figure family has also been regenerated and pinned from the
fixed script:
- `examples/publication_figures/artifacts/lhd_collisionality_summary.json`
- `docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality.png`
- `examples/publication_figures/artifacts/w7x_collisionality_summary.json`
- `docs/_static/figures/paper/sfincs_jax_fig2_w7x_collisionality.png`

The full artifacts are the validation-facing collisionality lane. The fast artifacts
remain useful as cheap branch-level regression scaffolds.

Autodiff and sensitivity validation:

```bash
python examples/publication_figures/generate_autodiff_sensitivity_validation.py
```

This writes a machine-readable summary plus PNG/PDF figures for:
- centered finite-difference agreement with JAX implicit-diff gradients,
- primal and adjoint residual gates for `custom_linear_solve`,
- solve-count scaling for implicit gradients versus finite differences,
- differentiable `geometryScheme=4` Boozer harmonic sensitivity maps.

Pinned outputs:
- `examples/publication_figures/artifacts/sfincs_jax_autodiff_sensitivity_validation_summary.json`
- `docs/_static/figures/paper/sfincs_jax_autodiff_gradient_check.png`
- `docs/_static/figures/paper/sfincs_jax_autodiff_sensitivity_map.png`

W7-X ambipolar validation scaffold:

```bash
python examples/publication_figures/generate_w7x_ambipolar_validation.py \
  --fast \
  --n-points 7 \
  --work-dir examples/publication_figures/output/w7x_ambipolar_validation_fast \
  --summary-json examples/publication_figures/output/w7x_ambipolar_validation_fast/summary.json \
  --out-dir docs/_static/figures/paper
```

This lane currently ships as an executable scaffold with a metadata-rich JSON summary
and figure writer, but it is not promoted to a checked-in W7-X literature artifact
until the heavier reference input is rerun and audited.

The ambipolar scaffold now supports restart and split execution as well:

```bash
python examples/publication_figures/generate_w7x_ambipolar_validation.py \
  --work-dir examples/publication_figures/output/w7x_ambipolar_validation_reference \
  --summary-json examples/publication_figures/output/w7x_ambipolar_validation_reference/summary.json \
  --out-dir docs/_static/figures/paper \
  --skip-existing \
  --scan-only \
  --index 0 \
  --stride 2
```

Launch a second process with ``--index 1 --stride 2`` to fill the other half of the
``E_r`` ladder. Then rerun once without ``--scan-only`` and with ``--skip-existing``
to reuse the finished scan points, write the ambipolar summary, and generate the figure.
