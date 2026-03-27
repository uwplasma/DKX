# sfincs_jax

`sfincs_jax` is a JAX implementation of SFINCS v3 that solves the same neoclassical drift-kinetic problem with matching normalizations, geometry conventions, and output format (`sfincsOutput.h5`).

It is designed for:

- high-performance runs on CPU/GPU,
- memory-efficient large solves,
- end-to-end differentiable workflows.

## Installation

Install from PyPI:

```bash
pip install sfincs_jax
```

Install from source:

```bash
git clone https://github.com/uwplasma/sfincs_jax.git
cd sfincs_jax
pip install .
```

Development install:

```bash
git clone https://github.com/uwplasma/sfincs_jax.git
cd sfincs_jax
pip install -e ".[dev]"
```

## Quick Start (Python)

Read a namelist, run `sfincs_jax`, write `sfincsOutput.h5`, and inspect results directly in memory:

```python
from pathlib import Path

from sfincs_jax.io import write_sfincs_jax_output_h5

input_namelist = Path("input.namelist")
out_path, results = write_sfincs_jax_output_h5(
    input_namelist=input_namelist,
    output_path=Path("sfincsOutput.h5"),
    return_results=True,
)

print("Wrote:", out_path)
print("Available datasets:", len(results))
print("Example key:", "particleFlux_vm_psiHat" in results)
```

`sfincs_jax write-output` and `write_sfincs_jax_output_h5(...)` use the fast explicit
solve path by default. Request the implicit/differentiable linear-solve path only when
you need it:

```python
write_sfincs_jax_output_h5(
    input_namelist=input_namelist,
    output_path=Path("sfincsOutput.h5"),
    differentiable=True,
)
```

## Executable (CLI)

You can run `sfincs_jax` from anywhere in your terminal. You do not need to be inside the repository folder.

Run an input file (default behavior, same invocation style as Fortran SFINCS):

```bash
sfincs_jax /path/to/input.namelist
```

Write output explicitly:

```bash
sfincs_jax write-output --input /path/to/input.namelist --out /path/to/sfincsOutput.h5
```

Compare two outputs:

```bash
sfincs_jax compare-h5 --a sfincsOutput_jax.h5 --b sfincsOutput_fortran.h5
```

Advanced CLI/solver options are documented in `docs/usage.rst` and `docs/performance_techniques.rst`.

## Historical Reduced-Suite Comparison (Fortran v3 vs sfincs_jax)

Reproduce the table:

```bash
python scripts/run_reduced_upstream_suite.py \
  --fortran-exe /path/to/sfincs \
  --reuse-fortran \
  --max-attempts 1 \
  --rtol 5e-4 \
  --atol 1e-9 \
  --jax-repeats 2
python scripts/generate_readme_reduced_suite_table.py
```

Artifacts:

- `tests/reduced_upstream_examples/suite_report.json`
- `tests/reduced_upstream_examples/suite_report_strict.json`
- `docs/_generated/reduced_upstream_suite_status.rst`
- `docs/_generated/reduced_upstream_suite_status_strict.rst`

This reduced table is a historical upstream-reference snapshot. The current authoritative
branch-state comparison for all examples, CPU/GPU runtimes, memory, and mismatch/error status
is the full fast explicit example-suite table in the section below.

<!-- BEGIN REDUCED_SUITE_TABLE -->
| Case | Fortran CPU(s) | sfincs_jax CPU(s) | sfincs_jax GPU(s) | Fortran CPU MB | sfincs_jax CPU MB | sfincs_jax GPU MB | Mismatches (practical/strict) | Print comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HSX_FPCollisions_DKESTrajectories | 0.596 | 5.348 | 9.281 | 146.6 | 2932.1 | 2224.0 | 0/192 (strict 0/192) | 9/9 |
| HSX_FPCollisions_fullTrajectories | 2.702 | 5.110 | 10.901 | 99.9 | 1870.4 | 1743.2 | 0/192 (strict 0/192) | 9/9 |
| HSX_PASCollisions_DKESTrajectories | 2.778 | 61.428 | parity_ok | 346.0 | 5508.7 | parity_ok | 0/192 (strict 0/192) | 9/9 |
| HSX_PASCollisions_fullTrajectories | 32.903 | 196.472 | max_attempts | 467.0 | 3053.3 | max_attempts | 0/192 (strict 3/192) | 9/9 |
| filteredW7XNetCDF_2species_magneticDrifts_noEr | 3.642 | 3.797 | 4.901 | 138.0 | 558.7 | 1435.8 | 0/192 (strict 0/192) | 9/9 |
| filteredW7XNetCDF_2species_magneticDrifts_withEr | 4.337 | 3.782 | 5.457 | 137.5 | 595.7 | 1456.1 | 0/192 (strict 0/192) | 9/9 |
| filteredW7XNetCDF_2species_noEr | 5.021 | 2.761 | 3.993 | 136.0 | 819.5 | 1416.1 | 0/192 (strict 0/192) | 9/9 |
| geometryScheme4_1species_PAS_withEr_DKESTrajectories | 8.257 | 2.674 | max_attempts | 484.6 | 994.9 | max_attempts | 0/207 (strict 0/207) | 9/9 |
| geometryScheme4_2species_PAS_noEr | 0.355 | 3.498 | 7.063 | 139.0 | 878.9 | 1683.3 | 0/207 (strict 0/207) | 9/9 |
| geometryScheme4_2species_noEr | 0.300 | 3.475 | 6.010 | 134.5 | 1582.5 | 1999.0 | 0/206 (strict 0/206) | 9/9 |
| geometryScheme4_2species_noEr_withPhi1InDKE | 0.123 | 2.698 | 4.442 | 129.6 | 482.8 | 1417.9 | 0/264 (strict 0/264) | 9/9 |
| geometryScheme4_2species_noEr_withQN | 0.062 | 2.335 | 3.737 | 112.3 | 456.1 | 1400.2 | 0/264 (strict 0/264) | 9/9 |
| geometryScheme4_2species_withEr_fullTrajectories | 0.074 | 2.847 | 4.143 | 118.4 | 792.3 | 1417.6 | 0/192 (strict 0/192) | 9/9 |
| geometryScheme4_2species_withEr_fullTrajectories_withQN | 0.080 | 3.099 | 4.345 | 117.0 | 589.6 | 1438.4 | 0/250 (strict 0/250) | 9/9 |
| geometryScheme5_3species_loRes | 1.734 | 4.378 | 9.824 | 163.1 | 1648.6 | 1845.9 | 0/192 (strict 0/192) | 9/9 |
| inductiveE_noEr | 0.167 | 2.696 | 3.742 | 129.0 | 827.0 | 1439.3 | 0/206 (strict 0/206) | 9/9 |
| monoenergetic_geometryScheme1 | 0.612 | 4.465 | jax_error | 133.7 | 281.4 | jax_error | 0/203 (strict 4/203) | 7/9 |
| monoenergetic_geometryScheme11 | 2.665 | 6.405 | jax_error | 204.3 | 300.1 | jax_error | 0/207 (strict 0/207) | 7/9 |
| monoenergetic_geometryScheme5_ASCII | 0.978 | 5.228 | jax_error | 158.8 | 299.2 | jax_error | 0/205 (strict 2/206) | 7/9 |
| monoenergetic_geometryScheme5_netCDF | 0.971 | 5.048 | jax_error | 151.1 | 308.2 | jax_error | 0/205 (strict 2/206) | 7/9 |
| quick_2species_FPCollisions_noEr | 0.315 | 2.513 | 4.095 | 125.4 | 838.7 | 1438.6 | 0/206 (strict 0/206) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories | 0.065 | 1.926 | 3.743 | 113.9 | 598.8 | 1410.2 | 0/207 (strict 0/207) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories | 0.173 | 3.077 | 4.649 | 126.8 | 879.4 | 1476.4 | 0/206 (strict 0/206) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories | 2.452 | 22.427 | 218.818 | 263.1 | 2407.6 | 2382.4 | 0/206 (strict 0/206) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories | 4.769 | 8.444 | 59.438 | 375.5 | 1947.9 | 2374.1 | 0/206 (strict 0/206) | 9/9 |
| tokamak_1species_FPCollisions_noEr | 9.724 | 1.934 | 3.592 | 132.8 | 732.0 | 1386.7 | 0/187 (strict 12/187) | 9/9 |
| tokamak_1species_FPCollisions_noEr_withPhi1InDKE | 13.340 | 2.313 | 4.496 | 141.0 | 468.3 | 1410.7 | 0/274 (strict 0/274) | 9/9 |
| tokamak_1species_FPCollisions_noEr_withQN | 6.150 | 2.238 | 3.641 | 127.1 | 520.7 | 1415.7 | 0/274 (strict 0/274) | 9/9 |
| tokamak_1species_FPCollisions_withEr_DKESTrajectories | 4.400 | 2.062 | 3.239 | 116.8 | 527.0 | 1409.7 | 0/213 (strict 0/213) | 9/9 |
| tokamak_1species_FPCollisions_withEr_fullTrajectories | 55.728 | 4.194 | 8.477 | 334.2 | 1759.9 | 2046.6 | 0/142 (strict 0/142) | 7/7 |
| tokamak_1species_PASCollisions_noEr | 2.301 | 2.695 | 30.843 | 718.5 | 560.4 | 1684.3 | 0/140 (strict 0/140) | 7/7 |
| tokamak_1species_PASCollisions_noEr_Nx1 | 2.124 | 39.621 | 81.066 | 250.9 | 5260.5 | 3227.2 | 0/212 (strict 33/212) | 9/9 |
| tokamak_1species_PASCollisions_noEr_withQN | 4.851 | 106.650 | 127.487 | 389.6 | 556.9 | 1987.1 | 0/274 (strict 0/274) | 9/9 |
| tokamak_1species_PASCollisions_withEr_fullTrajectories | 49.530 | 11.348 | max_attempts | 574.4 | 1302.0 | max_attempts | 0/212 (strict 0/212) | 9/9 |
| tokamak_2species_PASCollisions_noEr | 5.667 | 3.922 | 35.621 | 478.3 | 3372.4 | 2465.3 | 0/212 (strict 0/212) | 9/9 |
| tokamak_2species_PASCollisions_withEr_fullTrajectories | 15.875 | 180.320 | 166.946 | 442.1 | 1732.1 | 1843.6 | 0/212 (strict 1/212) | 9/9 |
| transportMatrix_geometryScheme11 | 0.303 | 3.584 | jax_error | 129.2 | 256.3 | jax_error | 0/193 (strict 0/193) | 7/9 |
| transportMatrix_geometryScheme2 | 0.236 | 3.579 | jax_error | 118.8 | 251.6 | jax_error | 0/193 (strict 0/193) | 7/9 |
<!-- END REDUCED_SUITE_TABLE -->

Status labels in table cells:

- `max_attempts`: the suite runner retried/rescaled this case up to `--max-attempts` and still did not complete a successful comparison run.
- `jax_error`: the JAX run exited with an exception for that benchmark lane/case.

## Fast Explicit Branch Audit

Regenerate this block on the fast-path branch with:

```bash
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/scaled_example_suite_fast_cpu_rtwindow_v1 \
  --scale-factor 1.0 \
  --runtime-target-basis fortran \
  --fortran-min-runtime-s 1.0 \
  --fortran-max-runtime-s 20.0 \
  --runtime-adjustment-iters 3
python scripts/generate_readme_fast_branch_audit.py \
  --out-root tests/scaled_example_suite_fast_cpu_rtwindow_v1
```

The benchmark policy on this branch is now:

- start from the original Fortran v3 example resolution,
- only downscale when a case is too expensive for a practical suite run,
- benchmark JAX CPU and GPU against a frozen CPU-generated Fortran reference root,
- and never intentionally push a reduced case below about `1s` of Fortran wall time unless
  the original example is already that small.

That avoids the misleading sub-second Fortran rows that came from blind global downscaling,
keeps the GPU lane tied to a deterministic reference, and makes the additional example part
of the same artifact set as the standard suite.

<!-- BEGIN FAST_BRANCH_AUDIT -->
Current fast explicit CPU audit comes from `tests/scaled_example_suite_fast_cpu_full_v6_merged`.
Matching frozen-reference GPU audit comes from `tests/scaled_example_suite_fast_gpu_full_v8`.

- Recorded cases: `39/39`
- Practical status counts: `parity_ok=39`
- Strict status counts: `parity_ok=39`
- GPU practical status counts: `parity_ok=39`
- GPU strict status counts: `parity_ok=39`
- Remaining cases: none
- Additional example: `parity_ok` on CPU and `parity_ok` on GPU

Top CPU runtime offenders:
- `tokamak_1species_PASCollisions_withEr_fullTrajectories`: jax=37.747s fortran=0.017s ratio=2220.43x status=parity_ok, res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14}
- `HSX_PASCollisions_DKESTrajectories`: jax=4.900s fortran=0.994s ratio=4.93x status=parity_ok, res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20}
- `HSX_PASCollisions_fullTrajectories`: jax=4.563s fortran=2.510s ratio=1.82x status=parity_ok, res={'NTHETA': 6, 'NZETA': 15, 'NX': 3, 'NXI': 20}
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories`: jax=4.550s fortran=1.104s ratio=4.12x status=parity_ok, res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20}
- `geometryScheme4_2species_PAS_noEr`: jax=3.685s fortran=0.953s ratio=3.87x status=parity_ok, res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25}

Top CPU memory offenders:
- `monoenergetic_geometryScheme5_ASCII`: jax=2773.9 MB fortran=142.1 MB ratio=19.52x status=parity_ok, res={'NTHETA': 10, 'NZETA': 20, 'NX': 1, 'NXI': 16}
- `geometryScheme4_2species_PAS_noEr`: jax=2623.4 MB fortran=162.7 MB ratio=16.12x status=parity_ok, res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25}
- `HSX_PASCollisions_DKESTrajectories`: jax=2128.6 MB fortran=112.0 MB ratio=19.00x status=parity_ok, res={'NTHETA': 5, 'NZETA': 15, 'NX': 2, 'NXI': 20}
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`: jax=2075.7 MB fortran=144.6 MB ratio=14.36x status=parity_ok, res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20}
- `tokamak_2species_PASCollisions_noEr`: jax=1940.7 MB fortran=123.6 MB ratio=15.70x status=parity_ok, res={'NTHETA': 19, 'NZETA': 1, 'NX': 7, 'NXI': 39}

Top GPU runtime offenders:
- `geometryScheme5_3species_loRes`: jax=144.597s fortran=98.976s ratio=1.46x status=parity_ok, res={'NTHETA': 5, 'NZETA': 5, 'NX': 2, 'NXI': 4}
- `tokamak_1species_PASCollisions_withEr_fullTrajectories`: jax=87.134s fortran=0.017s ratio=5125.56x status=parity_ok, res={'NTHETA': 10, 'NZETA': 1, 'NX': 3, 'NXI': 14}
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`: jax=58.198s fortran=1.706s ratio=34.11x status=parity_ok, res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20}
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories`: jax=25.291s fortran=1.104s ratio=22.91x status=parity_ok, res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20}
- `monoenergetic_geometryScheme5_ASCII`: jax=17.433s fortran=1.052s ratio=16.57x status=parity_ok, res={'NTHETA': 10, 'NZETA': 20, 'NX': 1, 'NXI': 16}

Top GPU memory offenders:
- `geometryScheme4_2species_PAS_noEr`: jax=2552.1 MB fortran=162.7 MB ratio=15.69x status=parity_ok, res={'NTHETA': 8, 'NZETA': 11, 'NX': 4, 'NXI': 25}
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories`: jax=2354.4 MB fortran=144.6 MB ratio=16.28x status=parity_ok, res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20}
- `HSX_PASCollisions_fullTrajectories`: jax=2105.3 MB fortran=179.2 MB ratio=11.75x status=parity_ok, res={'NTHETA': 6, 'NZETA': 15, 'NX': 3, 'NXI': 20}
- `tokamak_2species_PASCollisions_noEr`: jax=1702.6 MB fortran=123.6 MB ratio=13.78x status=parity_ok, res={'NTHETA': 19, 'NZETA': 1, 'NX': 7, 'NXI': 39}
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories`: jax=1671.2 MB fortran=130.7 MB ratio=12.79x status=parity_ok, res={'NTHETA': 6, 'NZETA': 19, 'NX': 2, 'NXI': 20}

Current mismatches:
- CPU practical mismatches: none
- CPU strict mismatches: none
- GPU practical/strict mismatches: none

Full per-case runtime / memory table:
| Case | Fortran CPU(s) | JAX CPU(s) | CPU x | JAX GPU(s) | GPU x | Fortran MB | JAX CPU MB | CPU MB x | JAX GPU MB | GPU MB x | CPU mismatch | GPU mismatch | CPU print | GPU print | CPU status | GPU status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `HSX_FPCollisions_DKESTrajectories` | 29.664 | 2.907 | 0.10x | 5.956 | 0.20x | 103.0 | 474.8 | 4.61x | 967.2 | 9.39x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `HSX_FPCollisions_fullTrajectories` | 88.504 | 2.882 | 0.03x | 6.609 | 0.07x | 100.8 | 496.3 | 4.92x | 972.7 | 9.65x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `HSX_PASCollisions_DKESTrajectories` | 0.994 | 4.900 | 4.93x | 11.242 | 11.31x | 112.0 | 2128.6 | 19.00x | 1488.3 | 13.28x | 0/123 (strict 0/123) | 0/123 (strict 0/123) | 7/7 | 7/7 | parity_ok | parity_ok |
| `HSX_PASCollisions_fullTrajectories` | 2.510 | 4.563 | 1.82x | 11.600 | 4.62x | 179.2 | 1662.3 | 9.28x | 2105.3 | 11.75x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `additional_examples` | 120.074 | 1.596 | 0.01x | 3.487 | 0.03x | 102.1 | 407.4 | 3.99x | 930.1 | 9.11x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_magneticDrifts_noEr` | 89.052 | 1.816 | 0.02x | 4.045 | 0.05x | 103.2 | 475.3 | 4.60x | 949.3 | 9.19x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_magneticDrifts_withEr` | 95.440 | 1.910 | 0.02x | 4.246 | 0.04x | 96.2 | 516.7 | 5.37x | 958.8 | 9.97x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `filteredW7XNetCDF_2species_noEr` | 128.508 | 1.653 | 0.01x | 3.538 | 0.03x | 100.3 | 460.2 | 4.59x | 940.1 | 9.37x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_1species_PAS_withEr_DKESTrajectories` | 1.365 | 3.588 | 2.63x | 5.506 | 4.03x | 127.3 | 969.8 | 7.62x | 1307.5 | 10.27x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_PAS_noEr` | 0.953 | 3.685 | 3.87x | 9.286 | 9.74x | 162.7 | 2623.4 | 16.12x | 2552.1 | 15.69x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr` | 139.240 | 1.699 | 0.01x | 3.594 | 0.03x | 92.2 | 444.1 | 4.81x | 960.2 | 10.41x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr_withPhi1InDKE` | 293.275 | 1.973 | 0.01x | 4.647 | 0.02x | 100.6 | 468.6 | 4.66x | 990.8 | 9.84x | 0/264 (strict 0/264) | 0/264 (strict 0/264) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_noEr_withQN` | 146.734 | 1.661 | 0.01x | 3.944 | 0.03x | 95.1 | 452.5 | 4.76x | 975.3 | 10.26x | 0/264 (strict 0/264) | 0/264 (strict 0/264) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_withEr_fullTrajectories` | 58.053 | 1.749 | 0.03x | 4.146 | 0.07x | 113.4 | 463.4 | 4.09x | 958.1 | 8.45x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme4_2species_withEr_fullTrajectories_withQN` | 211.358 | 1.806 | 0.01x | 4.348 | 0.02x | 98.8 | 479.5 | 4.85x | 980.3 | 9.92x | 0/250 (strict 0/250) | 0/250 (strict 0/250) | 9/9 | 9/9 | parity_ok | parity_ok |
| `geometryScheme5_3species_loRes` | 98.976 | 1.750 | 0.02x | 144.597 | 1.46x | 129.6 | 540.1 | 4.17x | 1043.0 | 8.05x | 0/193 (strict 0/193) | 0/193 (strict 0/193) | 9/9 | 9/9 | parity_ok | parity_ok |
| `inductiveE_noEr` | 166.614 | 1.597 | 0.01x | 4.341 | 0.03x | 99.2 | 449.8 | 4.53x | 959.8 | 9.68x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme1` | 0.795 | 1.707 | 2.15x | 4.091 | 5.15x | 110.2 | 664.0 | 6.02x | 967.7 | 8.78x | 0/203 (strict 0/203) | 0/203 (strict 0/203) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme11` | 0.861 | 2.728 | 3.17x | 16.535 | 19.20x | 118.7 | 1164.6 | 9.81x | 1091.1 | 9.19x | 0/208 (strict 0/208) | 0/208 (strict 0/208) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme5_ASCII` | 1.052 | 2.653 | 2.52x | 17.433 | 16.57x | 142.1 | 2773.9 | 19.52x | 1296.0 | 9.12x | 0/205 (strict 0/205) | 0/205 (strict 0/205) | 9/9 | 9/9 | parity_ok | parity_ok |
| `monoenergetic_geometryScheme5_netCDF` | 1.029 | 2.133 | 2.07x | 14.875 | 14.46x | 131.4 | 1148.6 | 8.74x | 1073.7 | 8.17x | 0/205 (strict 0/205) | 0/205 (strict 0/205) | 9/9 | 9/9 | parity_ok | parity_ok |
| `quick_2species_FPCollisions_noEr` | 166.945 | 1.553 | 0.01x | 4.445 | 0.03x | 97.1 | 440.7 | 4.54x | 959.0 | 9.87x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories` | 76.666 | 1.754 | 0.02x | 3.998 | 0.05x | 106.7 | 464.1 | 4.35x | 964.2 | 9.03x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories` | 93.439 | 1.921 | 0.02x | 4.508 | 0.05x | 94.0 | 476.6 | 5.07x | 970.3 | 10.33x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories` | 1.104 | 4.550 | 4.12x | 25.291 | 22.91x | 130.7 | 874.1 | 6.69x | 1671.2 | 12.79x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories` | 1.706 | 3.440 | 2.02x | 58.198 | 34.11x | 144.6 | 2075.7 | 14.36x | 2354.4 | 16.28x | 0/207 (strict 0/207) | 0/207 (strict 0/207) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr` | 160.856 | 1.329 | 0.01x | 4.039 | 0.03x | 93.2 | 352.1 | 3.78x | 904.8 | 9.71x | 0/188 (strict 0/188) | 0/188 (strict 0/188) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr_withPhi1InDKE` | 259.575 | 1.930 | 0.01x | 5.406 | 0.02x | 89.6 | 441.9 | 4.93x | 980.8 | 10.95x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_noEr_withQN` | 237.879 | 1.546 | 0.01x | 4.244 | 0.02x | 102.6 | 406.7 | 3.96x | 963.2 | 9.39x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_withEr_DKESTrajectories` | 155.955 | 1.548 | 0.01x | 3.693 | 0.02x | 103.1 | 410.2 | 3.98x | 953.7 | 9.25x | 0/214 (strict 0/214) | 0/214 (strict 0/214) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_FPCollisions_withEr_fullTrajectories` | 154.953 | 1.878 | 0.01x | 3.741 | 0.02x | 101.1 | 421.0 | 4.16x | 960.9 | 9.51x | 0/214 (strict 0/214) | 0/214 (strict 0/214) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr` | 0.309 | 2.345 | 7.59x | 5.855 | 18.95x | 114.2 | 575.2 | 5.03x | 1028.2 | 9.00x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr_Nx1` | 0.017 | 1.753 | 103.14x | 5.610 | 330.01x | 100.9 | 481.9 | 4.77x | 971.1 | 9.62x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_noEr_withQN` | 0.888 | 1.986 | 2.24x | 4.347 | 4.90x | 120.9 | 496.9 | 4.11x | 1031.6 | 8.53x | 0/274 (strict 0/274) | 0/274 (strict 0/274) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_1species_PASCollisions_withEr_fullTrajectories` | 0.017 | 37.747 | 2220.43x | 87.134 | 5125.56x | 102.0 | 549.3 | 5.38x | 1076.8 | 10.55x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_2species_PASCollisions_noEr` | 0.331 | 3.555 | 10.74x | 12.356 | 37.33x | 123.6 | 1940.7 | 15.70x | 1702.6 | 13.78x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `tokamak_2species_PASCollisions_withEr_fullTrajectories` | 1.330 | 3.331 | 2.50x | 8.927 | 6.71x | 121.8 | 1586.3 | 13.02x | 1292.6 | 10.61x | 0/212 (strict 0/212) | 0/212 (strict 0/212) | 9/9 | 9/9 | parity_ok | parity_ok |
| `transportMatrix_geometryScheme11` | 0.025 | 1.605 | 64.20x | 3.795 | 151.82x | 102.6 | 405.2 | 3.95x | 968.7 | 9.44x | 0/194 (strict 0/194) | 0/194 (strict 0/194) | 9/9 | 9/9 | parity_ok | parity_ok |
| `transportMatrix_geometryScheme2` | 0.031 | 1.479 | 47.72x | 3.543 | 114.28x | 100.5 | 405.7 | 4.04x | 965.9 | 9.61x | 0/194 (strict 0/194) | 0/194 (strict 0/194) | 9/9 | 9/9 | parity_ok | parity_ok |
<!-- END FAST_BRANCH_AUDIT -->

## Documentation

Build docs locally:

```bash
sphinx-build -b html -W docs docs/_build/html
```

Entry points:

- `docs/index.rst`
- `docs/system_equations.rst`
- `docs/method.rst`
- `docs/normalizations.rst`
- `docs/performance.rst`
- `docs/parallelism.rst`

## Testing

```bash
pytest -q
```

## License

See `LICENSE`.
