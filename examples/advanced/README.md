# Advanced examples

Heavier scripts that build on the tutorial ladder (`../tutorials/`): shape
optimization through the full VMEX -> Boozer -> DKX chain, and the
high-fidelity promotion audit for completed `dkx scan-er` runs.  Most need the
optional `vmex` and `booz_xform_jax` packages and run for minutes to tens of
minutes; each has a smoke mode named in its docstring.

## Differentiable-kinetic optimization family

A simsopt-style family of single-script gradient optimizations that carry
`jax.grad` through the canonical kinetic solve. Each script keeps its input
parameters at the top, writes its objective inline, uses warm starts + GCROT
recycling across optimizer iterations, verifies autodiff against finite
differences, and saves a compressed before/after plot plus a history JSON.
Set `DKX_CI=1` to shrink resolution and iteration counts for a fast
smoke run.

- `examples/advanced/optimize_QA_bootstrap.py` — the flagship: a quasi-axisymmetric
  boundary optimized for low bootstrap current `<j.B>` through the full
  boundary -> vmex equilibrium -> Boozer -> kinetic-solve chain.
- `optimize_QH_bootstrap.py` — the quasi-helical analog on a precise-QH
  reactor-scale seed (nfp=4), following the vmex QH workflow.
- `optimize_electron_root.py` — shapes a Boozer `|B|` spectrum to steer the
  ambipolar radial electric field toward the electron root, differentiating
  through the ambipolar root (implicit function theorem, `dkx.er`).
- `optimize_impurity_screening.py` — shapes `|B|` to push a trace C6+ impurity
  flux outward (temperature screening) with the multi-species Fokker-Planck
  operator; also reports the temperature-screening coefficient from autodiff.
- `objectives.py` — a small shared library of composable `jax` figures of merit
  (bootstrap, particle/heat-flux L1/L2, impurity screening, ambipolar-root and
  quasisymmetry-residual metrics) plus two geometry/solve plumbing helpers.

These four scripts and `objectives.py` are exercised at CI resolution by the
optimization example test suite under the repository ``tests`` directory.

## VMEX optimization with a kinetic bootstrap current

`QA_optimization_bootstrap_dkx.py` is the flagship: VMEX's QA
bootstrap-current optimization example (`QA_optimization_bootstrap`) with the Redl bootstrap
row swapped for the drift-kinetic one DKX computes on each trial equilibrium.
The row is traced (boundary -> VMEX -> `booz_xform_jax` -> DKX), so VMEX's
implicit Jacobian carries it; `BOOTSTRAP_MODEL` selects `"dkx"`, `"redl"` or
`"both"`.  Adding the row to any other `vmex` optimization script is one
import and one tuple:

```python
from dkx.bootstrap import KineticBootstrapMismatch

kinetic = KineticBootstrapMismatch(profiles, surfaces=[0.25, 0.5, 0.75])
objective_function_terms.append((kinetic, 0.0, 1.0))
```

`profiles` is the same `vmex.core.bootstrap.KineticProfiles` the Redl term
takes, so both models describe one plasma, and the residual is Redl's
normalized mismatch, so the weights compare.  The default collision operator
is pitch-angle scattering, which has no momentum restoration and overestimates
`<j.B>`; `collision_operator=0` is Fokker-Planck.  Needs `vmex` and
`booz_xform_jax >= 0.4`; `DKX_EXAMPLES_CI=1` is a smoke pass.

At the committed parameters (three kinetic surfaces, `11 x 11 x 16 x 4`,
modes 1 then 2, ten evaluations each) one run on four laptop CPU threads took
30 minutes and 4.6 GB: objective 1.78 -> 0.0061, quasisymmetry 6.7e-2 ->
5.7e-3, and the DKX mismatch `f_boot` 1.2e-3 -> 1.2e-4 on the `ns = 101`
re-solve. The optimized equilibrium carries the pitch-angle-scattering
current, 1.5-1.6 times Redl's at `s = 0.25-0.75`: the operator's known excess,
not a physical difference.

![Bootstrap current before and after](../../docs/_static/figures/readme/QA_optimization_bootstrap_dkx.png)

`QH_optimization_bootstrap_dkx.py` (nfp=4) and `QI_optimization_bootstrap_dkx.py`
(nfp=2, where Redl, a fit to quasisymmetric calculations, is an extrapolation)
use the host term `KineticBootstrapCurrent` instead: DKX on the written wout,
under a finite-difference Jacobian.  Set `DKX_VMEX_ROOT` when `vmex` came from
a wheel rather than a checkout.

## Finite-beta VMEC to kinetic profiles

- `finite_beta_vmec_to_sfincs.py` — solves the finite-beta QA equilibrium in
  `input.nfp2_QA_finite_beta` with VMEX, then scans `E_r` with DKX on several
  surfaces and writes the ambipolar root, bootstrap current and flux profiles
  (the docs figure `finite_beta_vmex_sfincs_bootstrap_er.png`). Needs `vmex`;
  run with `--help` for the knobs.

## Proxy objectives and the promotion audit for completed `scan-er` runs

- `qa_nfp2_dkx_objectives.py` — fast differentiable JAX proxy lane for adding
  neoclassical objectives (bootstrap current, electron root, flux selectivity)
  to a QA optimization; writes the proxy-summary JSON the launcher reads.
  Accepted designs still need real `dkx scan-er` outputs before any claim.
- `evaluate_dkx_promotion_scan.py` — high-fidelity promotion audit for
  completed `dkx scan-er` directories. It reads `sfincsOutput.h5` files,
  checks ambipolar roots, bootstrap current, species fluxes, and residual gates,
  then writes JSON plus PNG/PDF promotion plots.
  Pass `--impurity-species-index` only for a real impurity/flux-selectivity
  objective; omit it for two-species ion/electron electron-root scans.
- `launch_dkx_candidate_scan.py` — accepted-candidate workflow from a
  proxy optimization JSON to a reproducible `dkx scan-er` command. By
  default it writes a JSON plan and prints commands; pass `--execute` only when
  ready to launch the high-fidelity scan.
- `compare_dkx_promotion_runs.py` — compares CPU/GPU promotion summaries
  and optional Fortran-v3 promotion summaries, writing JSON plus PNG/PDF
  comparison reports for selected ambipolar root, bootstrap objective, and flux
  objective gates. The docs include both a fast demo/format-only comparison and
  real reduced-W7-X and finite-beta QA comparisons generated from separate CPU,
  GPU, and SFINCS Fortran v3 promotion JSON files.

Real promotion checklist:

```bash
python examples/advanced/qa_nfp2_dkx_objectives.py --objective balanced --steps 120 --out-dir runs/qa_candidate01/proxy --stem candidate01_proxy
python examples/advanced/launch_dkx_candidate_scan.py --proxy-summary runs/qa_candidate01/proxy/candidate01_proxy.json --input runs/qa_candidate01/input_r0p50.namelist --out-dir runs/qa_candidate01/scan_cpu/r0p50 --er-min -3 --er-max 3 --n-er 7 --jobs 4
JAX_PLATFORM_NAME=cpu dkx scan-er --input runs/qa_candidate01/input_r0p50.namelist --out-dir runs/qa_candidate01/scan_cpu/r0p50 --values -3 -2 -1 0 1 2 3 --compute-solution --skip-existing --jobs 4
python examples/advanced/evaluate_dkx_promotion_scan.py --scan-dir runs/qa_candidate01/scan_cpu/r0p50 --out-dir runs/qa_candidate01/audit --stem candidate01_r0p50_cpu --require-electron-root
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORM_NAME=gpu dkx scan-er --input runs/qa_candidate01/input_r0p50.namelist --out-dir runs/qa_candidate01/scan_gpu/r0p50 --values -3 -2 -1 0 1 2 3 --compute-solution --skip-existing --jobs 1
python examples/advanced/evaluate_dkx_promotion_scan.py --scan-dir runs/qa_candidate01/scan_gpu/r0p50 --out-dir runs/qa_candidate01/audit --stem candidate01_r0p50_gpu --require-electron-root
python examples/advanced/compare_dkx_promotion_runs.py --cpu runs/qa_candidate01/audit/candidate01_r0p50_cpu.json --gpu runs/qa_candidate01/audit/candidate01_r0p50_gpu.json --out-dir runs/qa_candidate01/audit --stem candidate01_r0p50_comparison
```

Add `--fortran runs/qa_candidate01/audit/candidate01_r0p50_fortran.json` to the
final comparison only after a Fortran-v3-derived promotion audit has been
generated from matching completed scan points.

The no-impurity path is the default for two-species electron-root scans. Pass
`--impurity-species-index` only when a real impurity species is present and the
flux-selectivity objective is part of the claim.
