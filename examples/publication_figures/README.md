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

Prototype run used on this branch:

```bash
python examples/publication_figures/generate_er_trajectory_sweep.py \
  --preset tokamak_like \
  --fast \
  --er-values=-0.5,0.0,0.5 \
  --er-res 1.0 \
  --work-dir examples/publication_figures/output/er_sweep_fast_tokamak \
  --out-dir docs/_static/figures/paper
```

That bounded sweep produces:
- `examples/publication_figures/artifacts/er_sweep_fast_tokamak_summary.json`
- `docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep.png`
- `docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep.pdf`
