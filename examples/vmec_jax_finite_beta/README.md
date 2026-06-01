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
`NL=7` is already stable at `r_N=0.50`, but `Nxi=8` and `Nx=7` both affect the
root. Combined `Nxi=8,Nx=7` and `Nxi=9` probes were attempted on an RTX A4000
and are currently too expensive for this bounded documentation example.

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
  --ntheta 5 --nzeta 5 --nxi 7 --nl 4 --nx 5 \
  --real-ntheta 7 --real-nzeta 7 \
  --velocity-nxi 9 --velocity-nl 5 --velocity-nx 6
```

The `--with-errorbars` mode runs two additional bounded convergence probes.
The plotted bars are the pointwise maximum change in
`<J.B>/sqrt(<B^2>)` after separately refining angular real-space resolution
and velocity-space resolution from the baseline grid.

The default `--wout` is the reactor-scale Landreman-Paul QA reference because
SFINCS radial-coordinate conversions require a positive VMEC `Aminor_p`. If you
point the script at an unscaled wout with `Aminor_p = 0`, use `--skip-sfincs`
for Redl-only plotting or provide a physically scaled wout before running the
kinetic comparison.

If `vmec_jax` is installed from a source checkout, point the example at it:

```bash
export SFINCS_JAX_VMEC_JAX_PATH=/path/to/vmec_jax
```

Outputs are written under `examples/vmec_jax_finite_beta/output/` by default and
include `finite_beta_vmec_jax_sfincs_bootstrap_er.png`,
`finite_beta_vmec_jax_sfincs_bootstrap_er.pdf`,
`finite_beta_vmec_jax_sfincs_convergence_scan.png`,
`finite_beta_vmec_jax_sfincs_convergence_scan.pdf`, and JSON summaries.
