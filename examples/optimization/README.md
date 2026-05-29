# Optimization (Optax / JAX-native)

Optimization examples that leverage differentiability:
- fitting geometry harmonics
- calibrating parameters against frozen Fortran fixtures
- end-to-end objective optimization (with publication-style plots in some scripts)
- bounded optional ecosystem gates for differentiable objective wrappers
- QA nfp=2 neoclassical optimization proxies with explicit `sfincs_jax`
  high-fidelity promotion gates from completed `scan-er` outputs
- QA/QI NFP screening proxies that choose a QI fallback target when QA
  electron-root evidence is still under production resolution

Examples:
- `fit_geometry_harmonics_with_optax.py`
- `calibrate_nu_n_to_fortran_residual_fixture.py`
- `benchmark_optional_eqx_jaxopt_scheme4_gate.py` — optional Equinox/JAXopt gate on a real `geometryScheme=4` harmonic-fit objective; it verifies gradient agreement for an `equinox.Module` wrapper and, when JAXopt is installed explicitly, bounded loss reduction for `jaxopt.GradientDescent`. The JAXopt row skips cleanly in default CI.
- `qa_nfp2_sfincs_jax_objectives.py` — fast JAX proxy lane for adding
  neoclassical objectives to QA optimization. It supports bootstrap-current,
  electron-root, flux-selective, and balanced presets, writes JSON provenance,
  and generates PNG/PDF plots. The proxy layer is differentiable, but accepted
  designs still need real `sfincs_jax scan-er` outputs before kinetic
  validation or publication claims.
- `screen_qi_electron_root_nfp.py` — fast QA/QI screening lane for electron-root
  optimization fallback. It ranks QA and QI NFP candidates with the same
  proxy/evidence boundary, recommends QI `nfp=2` when QA remains deferred, and
  writes a promotion plan for the first real `scan-er` CPU/GPU/Fortran artifact.
- `evaluate_sfincs_jax_promotion_scan.py` — high-fidelity promotion audit for
  completed `sfincs_jax scan-er` directories. It reads `sfincsOutput.h5` files,
  checks ambipolar roots, bootstrap current, species fluxes, and residual gates,
  then writes JSON plus PNG/PDF promotion plots.
  Pass `--impurity-species-index` only for a real impurity/flux-selectivity
  objective; omit it for two-species ion/electron electron-root scans.
- `launch_sfincs_jax_candidate_scan.py` — accepted-candidate handoff from a
  proxy optimization JSON to a reproducible `sfincs_jax scan-er` command. By
  default it writes a JSON plan and prints commands; pass `--execute` only when
  ready to launch the high-fidelity scan.
- `materialize_qi_nfp2_promotion_input.py` — writes a low-resolution
  two-species QI `nfp=2` input derived from
  `examples/additional_examples/input.namelist` plus JSON provenance. This is
  only a kinetic promotion candidate; electron-root claims require completed
  CPU/GPU/Fortran scan and comparison gates. The checked docs now contain the
  first low-resolution CPU/GPU/Fortran artifact plus a refined `9 x 9 x 11 x 4`
  rung and a second `11 x 11 x 13 x 4` CPU/GPU/Fortran rung after the bounded
  RHSMode=1 dense-policy fix; the remaining root drift keeps the
  production-resolution ladder open.
- `compare_sfincs_jax_promotion_runs.py` — compares CPU/GPU promotion summaries
  and optional Fortran-v3 promotion summaries, writing JSON plus PNG/PDF
  comparison reports for selected ambipolar root, bootstrap objective, and flux
  objective gates. The docs include both a fast demo/format-only comparison and
  real reduced-W7-X and finite-beta QA comparisons generated from separate CPU,
  GPU, and SFINCS Fortran v3 promotion JSON files.
- `run_promotion_evidence_campaign.py` — one-command campaign wrapper that
  writes a JSON plan, runs selected CPU/GPU/Fortran scan lanes, audits each
  completed scan, and compares the resulting promotion JSON files. Use
  `--dry-run` first on expensive inputs. The Fortran lane allows missing
  linear-residual datasets by default because upstream v3 outputs often do not
  write the JAX residual fields; JAX CPU/GPU lanes still require residuals. For
  expensive GPU campaigns, add `--jax-scan-timeout-s` and
  `--promotion-timeout-s` so a stalled lane writes a fail-closed campaign JSON.
- `summarize_finite_beta_electron_root_ladder.py` — reads already-promoted
  finite-beta QA electron-root CPU/GPU/Fortran JSON files across resolution
  tiers, checks backend root agreement and root drift, and writes a fail-closed
  convergence-ladder JSON/PNG/PDF. It reports under-resolved ladders as
  `deferred`, not `pass`.

Real promotion checklist:

```bash
python examples/optimization/qa_nfp2_sfincs_jax_objectives.py --objective balanced --steps 120 --out-dir runs/qa_candidate01/proxy --stem candidate01_proxy
python examples/optimization/screen_qi_electron_root_nfp.py --steps 70 --out-dir runs/qa_candidate01/proxy --stem qi_electron_root_nfp_screen
python examples/optimization/launch_sfincs_jax_candidate_scan.py --proxy-summary runs/qa_candidate01/proxy/candidate01_proxy.json --input runs/qa_candidate01/input_r0p50.namelist --out-dir runs/qa_candidate01/scan_cpu/r0p50 --er-min -3 --er-max 3 --n-er 7 --jobs 4
python examples/optimization/run_promotion_evidence_campaign.py --input runs/qa_candidate01/input_r0p50.namelist --out-dir runs/qa_candidate01/evidence_r0p50 --values -3 -2 -1 0 1 2 3 --run-cpu --run-gpu --gpu-device 0 --run-fortran --fortran-exe /path/to/sfincs --jobs 4 --jax-scan-timeout-s 1800 --promotion-timeout-s 300 --dry-run
JAX_PLATFORM_NAME=cpu sfincs_jax scan-er --input runs/qa_candidate01/input_r0p50.namelist --out-dir runs/qa_candidate01/scan_cpu/r0p50 --values -3 -2 -1 0 1 2 3 --compute-solution --skip-existing --jobs 4
python examples/optimization/evaluate_sfincs_jax_promotion_scan.py --scan-dir runs/qa_candidate01/scan_cpu/r0p50 --out-dir runs/qa_candidate01/audit --stem candidate01_r0p50_cpu --require-electron-root
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORM_NAME=gpu sfincs_jax scan-er --input runs/qa_candidate01/input_r0p50.namelist --out-dir runs/qa_candidate01/scan_gpu/r0p50 --values -3 -2 -1 0 1 2 3 --compute-solution --skip-existing --jobs 1
python examples/optimization/evaluate_sfincs_jax_promotion_scan.py --scan-dir runs/qa_candidate01/scan_gpu/r0p50 --out-dir runs/qa_candidate01/audit --stem candidate01_r0p50_gpu --require-electron-root
python examples/optimization/compare_sfincs_jax_promotion_runs.py --cpu runs/qa_candidate01/audit/candidate01_r0p50_cpu.json --gpu runs/qa_candidate01/audit/candidate01_r0p50_gpu.json --out-dir runs/qa_candidate01/audit --stem candidate01_r0p50_comparison
```

Add `--fortran runs/qa_candidate01/audit/candidate01_r0p50_fortran.json` to the
final comparison only after a Fortran-v3-derived promotion audit has been
generated from matching completed scan points.

QI `nfp=2` low-resolution promotion input materialization:

```bash
python examples/optimization/materialize_qi_nfp2_promotion_input.py --out-dir runs/qi_nfp2_candidate01 --stem qi_nfp2_lowres
python examples/optimization/run_promotion_evidence_campaign.py --input runs/qi_nfp2_candidate01/qi_nfp2_lowres.input.namelist --out-dir runs/qi_nfp2_candidate01/evidence --values -0.3 -0.1 0 0.1 0.3 1 2 3 --run-cpu --run-gpu --run-fortran --jax-scan-timeout-s 1800 --promotion-timeout-s 300 --dry-run
```

This generated input is a kinetic promotion candidate only. Do not cite it as
electron-root evidence until the CPU/GPU/Fortran scans, promotion audits,
backend comparison, and resolution ladder have passed.
For this two-species ion/electron QI contract, leave the impurity objective
unset unless you explicitly add a third impurity species.

The no-impurity path is the default for two-species electron-root scans. Pass
`--impurity-species-index` only when a real impurity species is present and the
flux-selectivity objective is part of the claim.
