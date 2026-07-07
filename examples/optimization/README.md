# Optimization (Optax / JAX-native)

Optimization examples that leverage differentiability:
- QA nfp=2 neoclassical optimization proxies with explicit `sfincs_jax`
  high-fidelity promotion gates from completed `scan-er` outputs
- editable VMEC-JAX-style QA optimization with an optional bootstrap-current
  objective
- CPU/GPU/Fortran promotion audits for completed high-fidelity scans

Examples:
- `qa_nfp2_sfincs_jax_objectives.py` — fast JAX proxy lane for adding
  neoclassical objectives to QA optimization. It supports bootstrap-current,
  electron-root, flux-selective, and balanced presets, writes JSON provenance,
  and generates PNG/PDF plots. The proxy layer is differentiable, but accepted
  designs still need real `sfincs_jax scan-er` outputs before kinetic
  validation or publication claims.
- `qa_nfp2_bootstrap_current_comparison.py` — focused teaching and README
  figure generated from the real VMEC-JAX QA optimization output. It
  plots the VMEC QA nfp=2 LCFS, `|B|`, finite-iota profile, and VMEC
  `J.B/sqrt(B.B)` current diagnostic, with an optional overlay from a second
  `vmec_jax` result directory. This panel is an equilibrium diagnostic until
  promoted with completed `sfincs_jax scan-er` outputs.
- `QA_optimization_bootstrap_current.py` — editable `vmec_jax`-style QA
  optimization script with all knobs at the top. It is intentionally close to
  the VMEC-JAX QA optimization example, uses `MAX_MODE=3` for
  faster iteration, and adds an optional `JDotB` current objective controlled by
  `INCLUDE_BOOTSTRAP_CURRENT_OBJECTIVE`.
- `evaluate_sfincs_jax_promotion_scan.py` — high-fidelity promotion audit for
  completed `sfincs_jax scan-er` directories. It reads `sfincsOutput.h5` files,
  checks ambipolar roots, bootstrap current, species fluxes, and residual gates,
  then writes JSON plus PNG/PDF promotion plots.
  Pass `--impurity-species-index` only for a real impurity/flux-selectivity
  objective; omit it for two-species ion/electron electron-root scans.
- `launch_sfincs_jax_candidate_scan.py` — accepted-candidate workflow from a
  proxy optimization JSON to a reproducible `sfincs_jax scan-er` command. By
  default it writes a JSON plan and prints commands; pass `--execute` only when
  ready to launch the high-fidelity scan.
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

The no-impurity path is the default for two-species electron-root scans. Pass
`--impurity-species-index` only when a real impurity species is present and the
flux-selectivity objective is part of the claim.
