# Finite-Beta VMEC-JAX to SFINCS-JAX

This directory contains a single runnable example for a finite-beta equilibrium:

```bash
python examples/vmec_jax_finite_beta/finite_beta_vmec_to_sfincs.py
```

The script runs the bundled `input.nfp2_QA_finite_beta` deck with `vmec_jax`,
writes a VMEC-style `wout` file, uses that file directly in `sfincs_jax`
`geometryScheme=5`, scans normalized `Er` on several flux surfaces, and plots
the ambipolar radial electric-field profile, the bootstrap-current radial
profile, representative radial-current and flux scans in `Er`, and a `jet`
contour plot of the sampled magnetic-field strength. The radial-profile x-axis
is normalized toroidal flux, `psi_N = r_N^2`.

This is a primal finite-beta transport example. It uses `vmec_jax` to generate
VMEC-style equilibrium data, then hands a `wout` file to `sfincs_jax`; it does
not differentiate through VMEC-JAX setup, file I/O, scheme-5 geometry evaluation,
the SFINCS kinetic solve, or radial-profile postprocessing. The summary JSON
records this as a workflow contract and includes radial-profile provenance:
requested `r_N` surfaces, plotted `psi_N = r_N^2` values, all bracketed roots,
the selected branch, and the convergence-overlay status.

The default radial scan uses `r_N = 0.15, 0.30, 0.50, 0.70, 0.85` and
root-neighborhood `Er = -9, -7, -5, -3, -1`, avoiding the singular magnetic axis
and exact VMEC boundary while resolving the ambipolar branch. The checked
documentation panel uses `Ntheta=7`, `Nzeta=7`, `Nxi=8`, `NL=6`, and `Nx=6`,
with adaptive midpoint refinement of bracketed ambipolar roots until the local
`Er` bracket is no wider than `1.25`. The convergence overlay uses the same
kinetic grid but refines every plotted surface to a local `Er` bracket width of
`0.625`. The default convergence gate is `max |Delta Er| <= 0.1` and
`max |Delta Jbs| <= 5e-4`; the checked documentation figure passes with
`max |Delta Er| = 2.1e-4` and `max |Delta Jbs| = 6.9e-7`. Use `--quick` for the
smoke-test resolution, or increase `--r-n-values`, `--er-values`, and the
resolution flags for production-quality profiles.

The separate convergence-scan plot summarizes the numerical sensitivity of the
finite-beta example:

```bash
python examples/vmec_jax_finite_beta/plot_convergence_scan.py
```

For the cached documentation campaign, kinetic-space refinement from
`7/6/6` to `8/6/6` still moves the ambipolar root, while the same-grid
root-bracket refinement at `8/6/6` is tight. Single-parameter probes show that
`NL=7` is stable at `r_N=0.50`, but `Nxi=8` and `Nx=7` both affect the
root. Combined `Nxi=8,Nx=7` and `Nxi=9` probes were attempted on an RTX A4000
and remain too expensive for this bounded documentation example.

The Landreman-Paul QA Redl comparison script is a separate diagnostic file, not
an optimizer. It evaluates the Redl bootstrap-current formula with `vmec_jax` on
the reactor-scale Landreman-Paul QA example and can optionally run `sfincs_jax`
on the same surfaces to compare
`FSABjHatOverRootFSAB2 * e n_bar sqrt(2 T_bar / m_bar)` against
`<J.B>/sqrt(<B^2>)` from the Redl fit:

```bash
python examples/vmec_jax_finite_beta/compare_landreman_paul_qa_bootstrap_redl.py --skip-sfincs
python examples/vmec_jax_finite_beta/compare_landreman_paul_qa_bootstrap_redl.py \
  --run-sfincs --with-errorbars \
  --r-n-values 0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --n-lambda 16 \
  --ntheta 13 --nzeta 13 --nxi 13 --nl 13 --nx 13 \
  --real-ntheta 15 --real-nzeta 15 \
  --velocity-nxi 15 --velocity-nl 14 --velocity-nx 14 \
  --solver-tolerance 1e-6
```

The `--with-errorbars` mode runs two additional bounded convergence probes.
The plotted bars are the pointwise maximum change in
`<J.B>/sqrt(<B^2>)` after separately refining angular real-space resolution
and velocity-space resolution from the `13 x 13 x 13 x 13 x 13` baseline grid.

The default `--wout` is the reactor-scale Landreman-Paul QA reference because
SFINCS radial-coordinate conversions require a positive VMEC `Aminor_p`. If you
point the script at an unscaled wout with `Aminor_p = 0`, use `--skip-sfincs`
for Redl-only plotting or provide a physically scaled wout before running the
kinetic comparison.

The QS-paper comparison script uses the arXiv:2205.02914 Zenodo benchmark decks
to compare a whole-radius `sfincs_jax` RHSMode=1 scan with the Redl bootstrap
formula. For a user-facing QA or QH `sfincs_jax` versus Redl plot that does not
load or require SFINCS Fortran v3 outputs, use `--jax-vs-redl`:

```bash
python examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py \
  --case QA \
  --quick \
  --jax-vs-redl \
  --solve-method auto \
  --stem qs_paper_qa_jax_redl_quick

python examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py \
  --case QH \
  --quick \
  --jax-vs-redl \
  --solve-method auto \
  --stem qs_paper_qh_jax_redl_quick
```

This mode still needs the Zenodo input decks and VMEC `wout` file, plus
`vmec_jax` for the Redl algebra, but it does not need a SFINCS Fortran v3
executable or archived `psiN_*/sfincsOutput.h5` files.

If the archived SFINCS Fortran v3 `sfincsOutput.h5` files are present in the
Zenodo tree, they can be overlaid as a reference curve without installing or
running the Fortran executable. For this benchmark script, `auto` is run in the
runtime/non-autodiff lane (`SFINCS_JAX_IMPLICIT_SOLVE=0`) with the direct-tail
field-split Schur preconditioner and a bounded `2048 MB` preconditioner cap
unless those environment variables are already set:

```bash
python examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py \
  --case QA \
  --s-values 0.1,0.15,0.25,0.3,0.45,0.5,0.6,0.7,0.75,0.85,0.9 \
  --ntheta 13 --nzeta 13 --nxi 21 --nx 5 \
  --with-errorbars \
  --real-ntheta 15 --real-nzeta 15 \
  --velocity-nxi 25 --velocity-nx 6 \
  --solver-tolerance 1e-6 \
  --solve-method auto
```

For a user-facing `sfincs_jax` versus Redl-only plot, add `--jax-vs-redl` (alias
`--hide-fortran`). The bounded grid above is meant for education, workflow
timing, and convergence triage; it is not a production parity grid. If you also
have a same-resolution SFINCS Fortran v3 rerun, pass it with
`--fortran-case-root` and keep `--require-same-resolution` so the script refuses
mixed-grid comparisons. Use
`--verbose-sfincs` or set `SFINCS_JAX_EXAMPLE_VERBOSE=1` for production-grid
reruns so phase, preconditioner, and Krylov progress are printed while setup is
running. Use
`--case QH --stem qs_paper_qh_same_resolution_11surface` for the quasi-helical
benchmark. Increase the resolution and inspect the JSON metrics before using
either profile quantitatively. The `--with-errorbars` mode adds one real-space
refinement and one velocity-space refinement and plots the pointwise maximum
change as the numerical error bar. The generated JSON stores the
`linearSolver*` diagnostics from each HDF5 file, including solver kind, true
residual, target, iterations, setup time, solve time, and sparse factor
estimates.

If `vmec_jax` is installed from a source checkout, point the example at it:

```bash
export SFINCS_JAX_VMEC_JAX_ROOT=/path/to/vmec_jax
```

Outputs are written under `examples/vmec_jax_finite_beta/output/` by default and
include `finite_beta_vmec_jax_sfincs_bootstrap_er.png`,
`finite_beta_vmec_jax_sfincs_bootstrap_er.pdf`,
`finite_beta_vmec_jax_sfincs_convergence_scan.png`,
`finite_beta_vmec_jax_sfincs_convergence_scan.pdf`, and JSON summaries.
