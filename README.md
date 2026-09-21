# DKX

[![PyPI](https://img.shields.io/pypi/v/dkx)](https://pypi.org/project/dkx/)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/DKX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/DKX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/sfincs-jax?label=docs)](https://sfincs-jax.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/uwplasma/DKX)](LICENSE)

**Differentiable neoclassical transport for stellarators and tokamaks, in JAX.**

DKX solves the radially local, linearized drift-kinetic equation on a flux surface
and returns particle and heat fluxes, parallel flows, bootstrap current, transport
matrices and ambipolar radial electric fields. Its compatibility interface covers
SFINCS Fortran v3 physics including full Fokker–Planck collisions and Phi1, and it
reads and writes SFINCS decks. Native workflows, CPU/GPU execution, and derivatives
have narrower qualified domains; see the [capability record](docs/capabilities.rst).

![W7-X standard configuration: |B| and parallel current density on the boundary, bootstrap current profile, ambipolar Er against Pablant et al. 2018](docs/_static/figures/readme/w7x_showcase.png)

## Install

```bash
pip install dkx                    # CPU
pip install -U "jax[cuda12]"       # add for NVIDIA GPUs
```

## Run

```python
import dkx

case = dkx.Case.from_mapping({           # analytic tokamak, teaching grid: seconds, not converged
    "schema": 1, "name": "tokamak", "run": {"workflow": "profile", "progress": False},
    "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.16, 0.25, 0.36]},
    "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                 "density_m3": [8.0e19, 7.0e19, 6.0e19], "temperature_keV": [1.0, 0.8, 0.6]}],
    "physics": {"model": "full_local", "collisions": "pitch_angle_scattering", "magnetic_drifts": "dkes", "phi1": "off"},
    "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
    "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}, "solver": {"method": "auto", "relative_tolerance": 1e-8},
})
result = dkx.run(case)
print("solver route:", result.metadata["solver_route"])
print("particle flux:", float(result.arrays["particle_flux_m2_s"][1, 0]))
```

From the shell: `dkx run case.toml`, `dkx converge case.toml` (check the resolution before
trusting a number), `dkx wout_w7x.nc`, or `dkx input.namelist` for a SFINCS deck.

## Gradients

```python
import jax, jax.numpy as jnp, dkx

case = dkx.Case.from_file("examples/05_ambipolar_profile/case.toml")
problem = dkx.prepare_er_scan(case, surface_index=1)          # geometry, grids, collisions once

def bootstrap_current(er_kv_m):
    scan = dkx.batched_er_scan(problem, er_kv_m, differentiable=True, retain_full_state=True)
    return jnp.sum(scan.moments["FSABjHat"])

j, dj_der = jax.jit(jax.value_and_grad(bootstrap_current))(jnp.array([-0.2, 0.0, 0.2]))
```

This prepared-scan example differentiates its admitted moment through the linear
solve. Supported profile and geometry inputs are listed in
[differentiability](docs/differentiability.rst). An explicit sparse transpose solve
is a linear-algebra capability, not by itself an autodiff or exact-gradient claim.

## Choose a workflow

| Calculation | Start from |
| --- | --- |
| Tokamak profile with prescribed `E_r` | `examples/01_tokamak_profile` |
| Stellarator from VMEC or Boozer files | `examples/02_vmec_stellarator`, `examples/03_boozer_stellarator` |
| Monoenergetic transport scan | `examples/04_monoenergetic_scan` |
| Ambipolar roots and branch selection | `examples/05_ambipolar_profile` |
| Resolution study, gradient checks | `examples/06_convergence_certificate`, `examples/07_gradients` |
| Geometry sensitivity and descent (analytic proxy) | `examples/08_vmex_optimization` |
| Phi1 and impurities (expert path) | `examples/09_phi1_and_impurities` |

## What works

| Workflow | DKX scope |
| --- | --- |
| Native research path | analytic, VMEC and Boozer profiles; PAS/full-FP subsets; regular ambipolar roots |
| SFINCS compatibility | decks and HDF5; RHSMode 2/3; Phi1, Tangential magnetic drifts, `export_f` and `lasym` expert paths |
| Repeated calculations | transport matrices, prepared scans, guarded Krylov recycling and direct factor reuse |
| Derivatives | implicit gradients for named prepared inputs, moments and regular roots; finite-difference checks on named cases |
| Hardware | CPU and qualified single-device GPU workflows; experimental multi-device sharding |

Exact model, interface, derivative and validation scope: [capability record](docs/capabilities.rst).

## Verified

![DKX against SFINCS Fortran v3, MONKES and YANCC: scaled differences on matched full Fokker-Planck decks, and Beidler-normalized monoenergetic coefficients](docs/_static/figures/readme/cross_code_validation.png)

The archived 38-deck same-discretization regression has median scaled difference
4e-6, with admitted full Fokker–Planck rows reaching 1e-8. Three named
monoenergetic comparisons with MONKES and YANCC place four Beidler-normalized
coefficients within 6 percent and `D33` within 0.1 percent. These are scoped
cross-code checks, not universal model validation. Derivative coverage is limited
to the inputs and outputs in the [validation matrix](docs/validation_matrix.rst).

## Recorded performance

![Runtime and peak memory, DKX against SFINCS Fortran v3, on the 744k-unknown HSX PAS case](docs/_static/figures/readme/tier1_hsx_runtime_memory.png)

Archived `HSX_PASCollisions_DKESTrajectories`, RHSMode=1, **744,610 unknowns**.
CPU reference runs used Apple M4 and PETSc 3.23 / MUMPS 5.8.2; the GPU row is
a separate RTX A4000 measurement. DKX warm timings exclude compilation.

| Measurement | Recorded time | Peak RSS |
| --- | ---: | ---: |
| DKX M4 CPU, ramped pitch, warm | **27.2 s** | **0.93 GB** |
| DKX M4 CPU, uniform pitch, warm | 44.3 s | 1.16 GB |
| DKX RTX A4000 GPU, warm | 45.0 s | — |
| SFINCS v3 reference, 1 rank | 463.6 s | 3.98 GB |
| SFINCS v3 reference, 2 ranks | 229.5 s | 2.86 GB |

A separate M3 Max run measured **23.6 s cold / 20.0 s warm** on the large grid.

That is **one measured 744k-unknown HSX PAS case**, chosen for the structured
PAS route. The archived 38-deck campaign reports the structured route faster on
9 of 9 completed comparisons and the Krylov route faster on 7 of 23, with six
incomplete cases. It does not establish a universal speed ordering. Every deck,
hardware string and method: [performance](docs/performance.rst).

![Measured parity envelopes of DKX against SFINCS Fortran v3](docs/_static/figures/readme/canonical_parity.png)

## Documentation

[Tutorial: first W7-X result](docs/usage.rst) ·
[How-to: resolution, wout files, SFINCS decks](docs/case_files.rst) ·
[Reference: schema, CLI, API](docs/api.rst) ·
[Explanation: physics models, solver routes, limitations](docs/physics_models.rst)

## Cite

```bibtex
@software{dkx,
  author = {Jorge, Rogerio and contributors},
  title  = {DKX: differentiable drift-kinetic neoclassical transport in JAX},
  url    = {https://github.com/uwplasma/DKX},
  year   = {2026}
}
```

Please also cite SFINCS (Landreman, Smith, Mollén & Helander, Phys. Plasmas 21, 042503,
2014) when you use its decks or model. Metadata: [CITATION.cff](CITATION.cff).

[Examples](examples/) · [Research plan](plan.md) · [Contributing](docs/contributing.rst) ·
[SOLVAX](https://github.com/uwplasma/SOLVAX) owns the reusable solver algorithms · [LICENSE](LICENSE)
