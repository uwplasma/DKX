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
- `generate_autodiff_sensitivity_validation.py`
- `generate_w7x_high_nu_performance.py`

Reference trajectory-sweep figure commands:

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

The corresponding checked summaries and figures are:
- `examples/publication_figures/artifacts/er_sweep_tokamak_reference_summary.json`
- `examples/publication_figures/artifacts/er_sweep_stellarator_fast_reference_summary.json`
- `docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep_tokamak_reference.png`
- `docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep_stellarator_fast_reference.png`

Bounded collisionality regression artifact:

```bash
python examples/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --fast \
  --scan-only \
  --work-dir examples/publication_figures/output/lhd_reaudit_fast
```

`generate_sfincs_paper_figs.py` writes machine-readable collisionality summaries
to `--summary-dir` as well. If `--summary-dir` is omitted, the summaries are written
into the selected `--work-dir` with top-level `metadata` and sorted `rows`.

For the heavy full-resolution re-audit lanes, the generator supports
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

The bounded LHD regression artifact is checked in as:
- `examples/publication_figures/artifacts/lhd_collisionality_reaudit_fast_summary.json`
- `docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality_reaudit_fast.png`

The bounded W7-X regression artifact is checked in as:
- `examples/publication_figures/artifacts/w7x_collisionality_reaudit_fast_summary.json`
- `docs/_static/figures/paper/sfincs_jax_fig2_w7x_collisionality_reaudit_fast.png`

The full collisionality figure family is checked in as:
- `examples/publication_figures/artifacts/lhd_collisionality_summary.json`
- `docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality.png`
- `examples/publication_figures/artifacts/w7x_collisionality_summary.json`
- `docs/_static/figures/paper/sfincs_jax_fig2_w7x_collisionality.png`

The full artifacts are the validation-facing collisionality lane. The bounded
artifacts are cheap regression inputs for branch-level checks.

Simakov-Helander high-collisionality audit artifacts:

This writes a bounded normalization/readiness audit for the full Appendix-B analytic
limit lane. It uses the checked-in LHD/W7-X collisionality summaries plus
representative `sfincsOutput.h5` geometry outputs to recompute `FSABHat2`, record the
available Appendix-B geometry quantities, fit inverse-`nu` FP tails, and keep the full
Simakov-Helander reproduction explicitly closed until wider high-`nu` scans are
pinned.

Pinned outputs:
- `examples/publication_figures/artifacts/sfincs_jax_simakov_helander_limit_audit_summary.json`
- `docs/_static/figures/paper/sfincs_jax_simakov_helander_limit_audit.png`
- `docs/_static/figures/paper/sfincs_jax_simakov_helander_limit_audit.pdf`
- `examples/publication_figures/artifacts/sfincs_jax_simakov_helander_high_nu_run_plan.json`

Launch high-`nu` pilots before widening to full FP/PAS scans using the retained
`generate_sfincs_paper_figs.py` scan driver:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python examples/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --collision-operators 0 \
  --nuprime-min 17.78279101649707 \
  --nuprime-max 17.78279101649707 \
  --n-points 1 \
  --timeout-s 900 \
  --transport-workers 2 \
  --transport-parallel-backend gpu \
  --transport-sparse-direct-max 30000 \
  --require-residuals \
  --max-transport-residual 1e-6 \
  --max-transport-relative-residual 1e-6 \
  --skip-existing \
  --scan-only
```

The launcher forces the explicit executable solve path for scans
(`SFINCS_JAX_IMPLICIT_SOLVE=0`) so high-collisionality transport can use sparse-LU
first attempts/rescues when Krylov residuals stall. The checked-in high-`nu'`
run plan uses a bounded sparse-direct cap and strict absolute/relative residual
gates. LHD FP is accepted only with clean residuals. W7-X FP high-`nu'` has
a residual-clean sparse-LU route with float32 host factors, exact matrix-free
residual verification, block-basis sparse-helper materialization, and
within-solve factor reuse across transport RHS solves. The checked performance
summary records the representative full-resolution W7-X point, its residual
diagnostics, and the solver provenance needed to reproduce the figure. Those
residual thresholds are also wired as fail-fast aborts for benchmark runs.

Generate the W7-X high-`nu'` performance figure from the checked summary:

```bash
python examples/publication_figures/generate_w7x_high_nu_performance.py
```

Pinned outputs:
- `examples/publication_figures/artifacts/sfincs_jax_w7x_high_nu_performance_summary.json`
- `docs/_static/figures/paper/sfincs_jax_w7x_high_nu_performance.png`
- `docs/_static/figures/paper/sfincs_jax_w7x_high_nu_performance.pdf`

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

W7-X ambipolar validation lane:

This lane is deferred until a defensible W7-X equilibrium/profile
reconstruction and matching checked-in source artifact are available. The stable
tree keeps the provenance template and package-level artifact gates:

- `examples/publication_figures/provenance/w7x_ambipolar_provenance_template.json`
- `sfincs_jax.validation.figures.build_w7x_ambipolar_root_provenance_panel`

Long W7-X ambipolar scan/figure generation belongs on the publication-audits
research branch. Stable tests keep the deferred panel fail-closed behavior,
provenance completeness checks, and checked-artifact admission gates.
