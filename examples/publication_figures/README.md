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
