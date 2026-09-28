# Kinetic objectives in stellarator optimization

Stellarator design optimizes the plasma boundary. A neoclassical objective, a
bootstrap current or a particle flux, depends on the boundary through a chain:
boundary → equilibrium (`vmex`) → Boozer transform (`booz_xform_jax`) →
drift-kinetic solve (DKX) → scalar. This tutorial covers the three levels at
which DKX joins that chain, from a self-contained shape derivative to a full
VMEX optimization, and states exactly what each level differentiates.

`vmex` and `booz_xform_jax` are optional. Nothing in DKX's default install or
CI requires them, and every script below either runs without them or reports
that they are missing.

## Level 1: a shape derivative without an equilibrium code

`examples/08_vmex_optimization` differentiates the kinetic solve with respect to
the geometry itself. The design variable is the amplitude of the helical
$\lvert B\rvert$ harmonic of a three-harmonic analytic surface ($N = 5$ field
periods, one hydrogen species, pitch-angle scattering). The traced chain is

$$
\epsilon_h \;\to\; \text{Boozer } \lvert B\rvert \text{ spectrum} \;\to\;
\texttt{FluxSurfaceGeometry.from\_fourier} \;\to\; K(\epsilon_h) \;\to\;
f \;\to\; \Gamma,
$$

and five steps of gradient descent reduce the radial particle flux. Only the
geometry leaves of the operator (`b_hat`, its angular derivatives, `d_hat` and
the covariant and contravariant field components) are replaced, so the
collision operator and grids stay fixed and the derivative is a pure shape
derivative. The gradient is checked against central differences.

```console
python examples/08_vmex_optimization/run.py
```

In a production loop the amplitudes are not typed by hand: `vmex` solves the
equilibrium and `booz_xform_jax` produces the spectrum. The script prints the
hand-off contract (below) rather than asserting more, and does not import
`vmex`.

## Level 2: the VMEC-to-Boozer proxy workflow

The hand-off between the equilibrium codes and DKX is a documented workflow
with a machine-checked scope. Start with the status preflight, which imports
neither optional backend:

```console
python examples/optimization/vmex_workflow_status.py --json
```

It reports whether `vmex` and `booz_xform_jax` are importable, runs a
no-dependency Boozer-spectrum autodiff readiness check, and prints the exact
command for the optional proxy-gradient check. The payload carries two
contracts:

- `no_solve_provenance_gate`: a machine-readable assertion that the workflow is
  a proxy-gradient path. `kinetic_solve_executed` is false, and the
  differentiated object is a Boozer-spectrum transport-like scalar.
- `kinetic_transport_scalar_contract`: the forward contract for a future
  VMEC/Boozer-to-kinetic-transport scalar. It lists
  `required_kinetic_transport_scalar_stages`: `vmec_source`,
  `vmec_equilibrium_or_wout`, `boozer_transform`, `sfincs_geometry_adapter`,
  `kinetic_operator_assembly`, `linear_kinetic_solve`,
  `transport_scalar_reduction`, `gradient_validation`. Each stage records its
  role, differentiability boundary, status and the evidence needed before it can
  support a kinetic transport scalar. The public scalar is
  `boozer_spectrum_proxy_not_kinetic`; its `no_overclaim_gate.status` must be
  `"pass"`, which fails if the proxy workflow claims a kinetic solve, requires
  optional packages in default CI, drops a stage, or promotes the kinetic scalar
  while stages remain deferred.

The end-to-end pipeline has the same skip-safe check:

```console
python examples/autodiff/vmex_to_boozer_sfincs_pipeline.py --check-backends --json
python examples/autodiff/vmex_to_boozer_sfincs_pipeline.py --check-backends --summary-json workflow-summary.json
```

With `--check-backends` it evaluates a small synthetic Boozer spectrum through
`dkx.workflows.geometry_adapters.boozer_spectrum_proxy_transport_objective`,
checks the full spectral gradient against centered finite differences and a
JVP against the gradient dot product, and reports `backend_readiness_gate`. This
is a readiness check of the downstream differentiable objective, not evidence
that `vmex` or `booz_xform_jax` ran. `--summary-json` writes the provenance
record.

With the backends installed (for example `python -m pip install -e` on local
checkouts of each), run the file-backed check on an explicit `wout`:

```console
python examples/autodiff/vmex_to_boozer_sfincs_pipeline.py \
  --wout wout_circular_tokamak.nc --mboz 3 --nboz 3 --surface 0.5 --steps 0 \
  --summary-json workflow-summary.json
```

or let `vmex` build the equilibrium first with `--vmec-case circular_tokamak
--vmec-max-iter 1`. The written summary must show
`no_solve_provenance_gate.status == "pass"`, which requires provenance for the
source `wout`, the surface, the Boozer resolution, the objective grid and the
spectral scale.

What this workflow differentiates: the scaled VMEC-like spectral arrays, the
`booz_xform_jax` transform, and the proxy objective. What it does not: VMEC
file I/O, the `vmex` fixed-boundary setup, and DKX's VMEC file readers, which
are setup only. Its explicit non-claims:

- no full VMEC-boundary-to-SFINCS kinetic transport gradients,
- no gradient through the kinetic transport solve in this workflow,
- no production solver dependency on `vmex` or `booz_xform_jax`.

The checks are `python -m pytest tests/test_vmex_workflow.py tests/test_jax_geometry_adapters.py -q`.
They include an invariant check that the normalized proxy objective is unchanged
by global scaling of the $\lvert B\rvert$ spectrum and is exactly zero, with
zero gradient, for constant $B$. Optional ecosystem benchmark tools are not part
of this workflow; external solver-library adoption studies are handled on research branches
until they meet the documented accuracy, runtime, memory and dependency
requirements.

## Level 3: a kinetic bootstrap row inside a VMEX optimization

`examples/optimization/QA_optimization_bootstrap_dkx.py` is VMEX's
self-consistent quasi-axisymmetric bootstrap-current optimization with the Redl
bootstrap row replaced by the drift-kinetic one DKX computes on each trial
equilibrium. The row, `dkx.bootstrap.KineticBootstrapMismatch`, is the
mismatch between the equilibrium's $\langle j\cdot B\rangle$ and the DKX kinetic
one, traced boundary → VMEX → `booz_xform_jax` → DKX so that VMEX's implicit
Jacobian carries it. Adding it to another `vmex` optimization script is one
import and one tuple:

```python
from dkx.bootstrap import KineticBootstrapMismatch

kinetic = KineticBootstrapMismatch(profiles, surfaces=[0.25, 0.5, 0.75])
objective_function_terms.append((kinetic, 0.0, 1.0))
```

`profiles` is the `vmex.core.bootstrap.KineticProfiles` object the Redl term
takes, so both models describe one plasma. `BOOTSTRAP_MODEL` in the script
selects `"dkx"`, `"redl"` or `"both"`. It needs `vmex` and
`booz_xform_jax >= 0.4`.

```{figure} ../_static/figures/readme/QA_optimization_bootstrap_dkx.png
:alt: Bootstrap current profiles and objective history of the VMEX QA optimization with a DKX kinetic row.
:width: 85%

At the committed parameters (three kinetic surfaces, $11 \times 11 \times 16
\times 4$, modes 1 then 2, ten evaluations each), one run on four laptop CPU
threads took 30 minutes and 4.6 GB. The objective fell from 1.78 to 0.0061,
quasisymmetry from 6.7e-2 to 5.7e-3, and the DKX mismatch from 1.2e-3 to
1.2e-4 (`examples/optimization/README.md`). Finite differences agree with the
traced Jacobian of the row to 5.7e-5–1.8e-3.
```

Two cautions go with this result. The default collision operator is
pitch-angle scattering, which lacks momentum restoration: the optimized
equilibrium carries 1.5–1.6 times Redl's current at $s = 0.25$–$0.75$, the
operator's known excess rather than a physical difference. Pass
`collision_operator=0` for full Fokker–Planck. And the kinetic grid is small,
so the kinetic current is not a converged value; check it at higher resolution
before quoting it.

`QH_optimization_bootstrap_dkx.py` (nfp = 4) and
`QI_optimization_bootstrap_dkx.py` (nfp = 2, where Redl is an extrapolation) use
the host-side term `KineticBootstrapCurrent` instead: DKX on the written `wout`,
under a finite-difference Jacobian.

## Other objectives

`examples/optimization/` also holds single-script optimizations that carry
`jax.grad` through the kinetic solve with warm starts and GCROT recycling across
iterations, each verified against finite differences:
`optimize_QA_bootstrap.py`, `optimize_QH_bootstrap.py`,
`optimize_electron_root.py` (steer the ambipolar $E_r$ toward the electron root,
differentiating through the root) and `optimize_impurity_screening.py` (push a
trace C6+ flux outward with the multi-species Fokker–Planck operator).
`objectives.py` collects the figures of merit. `DKX_CI=1` shrinks each to a
smoke run.

An optimized design is not validated by its proxy. The promotion scripts in the
same folder (`launch_dkx_candidate_scan.py`, `evaluate_dkx_promotion_scan.py`,
`compare_dkx_promotion_runs.py`) turn an accepted candidate into full `dkx
scan-er` runs on CPU and GPU and audit roots, bootstrap current, fluxes and
residuals before any claim is made.
