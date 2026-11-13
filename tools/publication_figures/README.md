# Publication-style figures

These scripts generate polished figures (PNG + PDF) intended to be *paper-ready*.

Requirements:

```bash
pip install matplotlib
```

Outputs are written under `tools/publication_figures/figures/` by default.
The literature-facing validation/figure map is tracked in
`tools/publication_figures/validation_manifest.json`.

Examples:
- `generate_sfincs_paper_figs.py`
- `generate_er_trajectory_sweep.py`
- `generate_autodiff_sensitivity_validation.py`
- `generate_w7x_high_nu_performance.py`
- `generate_readme_showcase.py` (the README's reference-limits and factor-reuse figures,
  from recorded measurements; writes to `docs/_static/figures/readme/`)

Reference trajectory-sweep figure commands:

```bash
python tools/publication_figures/generate_er_trajectory_sweep.py \
  --preset tokamak_like \
  --er-values=-30,0,30 \
  --work-dir tools/publication_figures/output/er_sweep_tokamak_reference \
  --summary-json tools/publication_figures/output/er_sweep_tokamak_reference/summary.json \
  --out-dir docs/_static/figures/paper \
  --stem dkx_er_trajectory_sweep_tokamak_reference

python tools/publication_figures/generate_er_trajectory_sweep.py \
  --preset stellarator_like \
  --fast \
  --er-values=-8.5897,0,8.5897 \
  --work-dir tools/publication_figures/output/er_sweep_stellarator_fast_reference \
  --summary-json tools/publication_figures/output/er_sweep_stellarator_fast_reference/summary.json \
  --out-dir docs/_static/figures/paper \
  --stem dkx_er_trajectory_sweep_stellarator_fast_reference
```

The corresponding checked summaries and figures are:
- `tools/publication_figures/artifacts/er_sweep_tokamak_reference_summary.json`
- `tools/publication_figures/artifacts/er_sweep_stellarator_fast_reference_summary.json`
- `docs/_static/figures/paper/dkx_er_trajectory_sweep_tokamak_reference.png`
- `docs/_static/figures/paper/dkx_er_trajectory_sweep_stellarator_fast_reference.png`

`generate_sfincs_paper_figs.py` writes machine-readable collisionality summaries
to `--summary-dir` as well. If `--summary-dir` is omitted, the summaries are written
into the selected `--work-dir` with top-level `metadata` and sorted `rows`.

For the heavy full-resolution re-audit lanes, the generator supports
split-operator execution so the FP and PAS ladders can be resumed independently
on separate devices:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python tools/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --collision-operators 0 \
  --skip-existing \
  --scan-only \
  --work-dir tools/publication_figures/output/lhd_reaudit_full \
  --summary-dir tools/publication_figures/artifacts

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python tools/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --collision-operators 1 \
  --skip-existing \
  --scan-only \
  --work-dir tools/publication_figures/output/lhd_reaudit_full \
  --summary-dir tools/publication_figures/artifacts
```

Once both operator ladders have completed, synthesize the audited summary and
figures without rerunning the scan:

```bash
python tools/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --plot-only \
  --collision-operators 0,1 \
  --work-dir tools/publication_figures/output/lhd_reaudit_full \
  --summary-dir tools/publication_figures/artifacts \
  --out-dir docs/_static/figures/paper
```

The same split-and-synthesize pattern applies to the W7-X collisionality lane.
When ``--skip-existing`` is used, the generator keeps completed scan points and
prunes only stale subdirectories that do not contain ``sfincsOutput.h5`` before
rerunning the missing points.

The full collisionality figure family is checked in as:
- `tools/publication_figures/artifacts/lhd_collisionality_summary.json`
- `docs/_static/figures/paper/dkx_fig1_lhd_collisionality.png`
- `tools/publication_figures/artifacts/w7x_collisionality_summary.json`
- `docs/_static/figures/paper/dkx_fig2_w7x_collisionality.png`

The full artifacts are the validation-facing collisionality lane. The bounded
artifacts are cheap regression inputs for branch-level checks.

Simakov-Helander high-collisionality limit:

The LHD and W7-X summaries above extend to `nu' = 100`. The audit uses them with
the Appendix-B geometry quantities recomputed from `sfincsOutput.h5` fields
(`FSABHat2` agrees to 1e-14) and compares the Fokker-Planck `L11` and `L12` with
the `C/nu'` asymptote at every scan point, fits the log-log tail slopes, and
records the gates read by the validation manifest.

Pinned outputs:
- `tools/publication_figures/artifacts/dkx_simakov_helander_limit_audit_summary.json`
- `docs/_static/figures/paper/dkx_fig3_simakov_helander.png`

The four-point-per-decade high-`nu'` extension (`nu' = 17.78` to `100`) was run per
case and per operator with residual gates on, then merged into the summaries:

```bash
python tools/publication_figures/generate_sfincs_paper_figs.py \
  --case lhd \
  --collision-operators 0 \
  --nuprime-min 17.78279101649707 \
  --nuprime-max 99.99998265800002 \
  --n-points 4 \
  --transport-sparse-direct-max 30000 \
  --require-residuals \
  --max-transport-residual 1e-6 \
  --max-transport-relative-residual 1e-6 \
  --skip-existing \
  --scan-only
```

The LHD PAS ladder at `nu' >= 17.8` needs `--solver-tolerance 1e-9`; at the
input's `1e-6` the Krylov stage stops at an absolute residual of 1.5e-6. Rebuild
Figures 1 to 3 from the checked-in summaries:

```bash
python tools/publication_figures/generate_sfincs_paper_figs.py --plot-summaries
```

The launcher forces the explicit executable solve path for scans
(`DKX_IMPLICIT_SOLVE=0`) so high-collisionality transport can use sparse-LU
first attempts/rescues when Krylov residuals stall. The high-`nu'`
scans use a bounded sparse-direct cap and strict absolute/relative residual
gates. LHD FP is accepted only with clean residuals. W7-X FP high-`nu'` has
a residual-clean sparse-LU route with float32 host factors, exact matrix-free
residual verification, block-basis sparse-helper materialization, and
within-solve factor reuse across transport RHS solves. The checked performance
summary records the representative full-resolution W7-X point, its residual
diagnostics, and the solver provenance needed to reproduce the figure. Those
residual thresholds are also wired as fail-fast aborts for benchmark runs.

Generate the W7-X high-`nu'` performance figure from the checked summary:

```bash
python tools/publication_figures/generate_w7x_high_nu_performance.py
```

Pinned outputs:
- `tools/publication_figures/artifacts/dkx_w7x_high_nu_performance_summary.json`
- `docs/_static/figures/paper/dkx_w7x_high_nu_performance.png`
- `docs/_static/figures/paper/dkx_w7x_high_nu_performance.pdf`

Autodiff and sensitivity validation:

```bash
python tools/publication_figures/generate_autodiff_sensitivity_validation.py
```

This writes a machine-readable summary plus PNG/PDF figures for:
- centered finite-difference agreement with JAX implicit-diff gradients,
- primal and adjoint residual gates for `custom_linear_solve`,
- solve-count scaling for implicit gradients versus finite differences,
- differentiable `geometryScheme=4` Boozer harmonic sensitivity maps.

Pinned outputs:
- `tools/publication_figures/artifacts/dkx_autodiff_sensitivity_validation_summary.json`
- `docs/_static/figures/paper/dkx_autodiff_gradient_check.png`
- `docs/_static/figures/paper/dkx_autodiff_sensitivity_map.png`
