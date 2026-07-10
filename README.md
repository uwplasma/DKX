# sfincs_jax

[![CI](https://github.com/uwplasma/sfincs_jax/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/uwplasma/sfincs_jax/actions/workflows/ci.yml)
[![Docs](https://github.com/uwplasma/sfincs_jax/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/uwplasma/sfincs_jax/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/sfincs_jax)](https://pypi.org/project/sfincs_jax/)
[![Coverage](https://codecov.io/gh/uwplasma/sfincs_jax/branch/main/graph/badge.svg)](https://codecov.io/gh/uwplasma/sfincs_jax)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/uwplasma/sfincs_jax)](LICENSE)

`sfincs_jax` solves the radially local, linearized drift-kinetic equation on a
flux surface — the same physics as [SFINCS Fortran v3](https://github.com/landreman/sfincs)
— in pure JAX. One `input.namelist` plus one geometry file gives neoclassical
particle/heat fluxes, parallel flows, bootstrap current, and transport matrices
for stellarators and tokamaks, on CPU or GPU, with end-to-end automatic
differentiation for sensitivities and optimization. Outputs, per-species result
tables, and console prints are pinned field-by-field against SFINCS Fortran v3.

## Installation

```bash
pip install sfincs_jax
```

Optional extras:

- **Structured direct solvers** (`solvax`, the external library that owns the
  block-tridiagonal Legendre elimination and recycled-Krylov tiers): until the
  `solvax` PyPI release, install it from git — the `[structured]` extra then
  resolves locally:

  ```bash
  pip install git+https://github.com/uwplasma/SOLVAX
  ```

  Without it, `sfincs_jax` imports lazily and falls back to host/direct paths.
- **GPU**: install the matching CUDA build of JAX, e.g.
  `pip install -U "jax[cuda12]"`.

Large public equilibrium files (W7-X, HSX) are not shipped in the wheel; they
are fetched from a GitHub release on first use and cached under
`~/.cache/sfincs_jax/data`. Prefetch with
`python -m sfincs_jax.validation.data_fetch`; see the
[installation docs](docs/installation.rst) for offline/cache options and for
building the SFINCS Fortran v3 reference executable with conda-provided
PETSc/MUMPS.

## Quickstart

Run a small circular-tokamak deck through the canonical driver (this mirrors
[`examples/run_tokamak.py`](examples/run_tokamak.py), which also builds the
namelist from Python dicts and plots the results):

```python
from pathlib import Path
from sfincs_jax.run import run_profile

deck = Path("input.namelist")
deck.write_text("""\
&geometryParameters
  geometryScheme = 1  ! circular tokamak: BHat = 1 + 0.1 cos(theta)
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0  GHat = 1.0  IHat = 0.0  iota = 1.31
  epsilon_t = 0.1  epsilon_h = 0.0  psiAHat = 0.045  aHat = 0.1
/
&speciesParameters
  Zs = 1  mHats = 1.0  nHats = 1.0  THats = 0.5
  dNHatdrHats = -6.0  dTHatdrHats = -3.0
/
&physicsParameters
  Delta = 4.5694d-3  alpha = 1.0  nu_n = 8.4774d-3
  Er = 0.0  collisionOperator = 1  ! pitch-angle scattering
/
&resolutionParameters
  Ntheta = 15  Nzeta = 1  Nxi = 8  NL = 4  Nx = 6
  solverTolerance = 1d-10
/
""")

run = run_profile(deck, solve_method="auto", out_path=Path("sfincsOutput.h5"))
print("particle flux:", float(run.moments["particleFlux_vm_psiHat"][0]))
print("bootstrap current <j.B>:", float(run.moments["FSABjHat"]))
```

`run_profile` prints the Fortran-parity console flow (banner, grids, solve
progress, per-species results table), writes `sfincsOutput.h5`/`.nc` keyed by
the SFINCS output names, and returns the state vector, solver statistics, and
all moments in memory. The CLI equivalent is
`sfincs_jax input.namelist --out sfincsOutput.h5`, and
`sfincs_jax --plot sfincsOutput.h5` builds a PDF diagnostics panel.

## Performance vs SFINCS Fortran v3

Measured head-to-head on the same machine (MacBook, Apple M4, 24 GB) and the
same deck: `HSX_PASCollisions_DKESTrajectories`, RHSMode=1, at
`Ntheta=25, Nzeta=51, Nxi=100, Nx=5` — 744,610 unknowns. The Fortran reference
is the conda PETSc 3.23 + MUMPS 5.8.2 build of SFINCS v3; `sfincs_jax` uses
the tier-1 truncated Legendre block elimination from `solvax`.

![Runtime and peak memory: sfincs_jax vs SFINCS Fortran v3 on the 744k-unknown HSX PAS case](docs/_static/figures/readme/tier1_hsx_runtime_memory.png)

- With the matched `Nxi`-for-`x` ramp discretization, `sfincs_jax` solves in
  **27.2 s at 0.93 GB** — 17x faster than 1-rank Fortran (463.6 s, 3.98 GB) and
  8.4x faster than Fortran's best measured parallel floor (229.5 s / 2.86 GB
  at 2 ranks; 4 and 8 ranks are slower on this machine), at roughly 30% of the
  memory. With uniform `Nxi` it takes 44.3 s at 1.16 GB; an RTX A4000 GPU
  takes 45.0 s (the Legendre scan is serial and A4000 FP64 is 1/32 rate).
- The Fortran strong-scaling baseline on the same case: 463.6 s (1 rank),
  229.5 s (2 ranks, 101% efficiency), 240.9 s (4 ranks), 270.5 s (8 ranks).
- At the full production resolution of this case (2.5 M unknowns), **neither**
  code fits a global sparse factorization in 24 GB; the truncated Legendre
  elimination needs only O(K m^2) memory (~0.3 GB here, vs ~91 GB for the
  full-band factor) and is the locally viable direct path.
- The direct solve is more converged than the Fortran reference: Fortran's own
  electron `FSABFlow` scatters 51% across its 1/2/4/8-rank runs (KSP
  rtol=1e-6 solver noise), while `sfincs_jax` matches the closest Fortran run
  to 2e-10 and sits inside Fortran's spread on every quantity.

Scope: this is one measured 744k-unknown HSX PAS case; further cases are
promoted here as each vertical slice lands with its own evidence. Regenerate
the figure from the recorded values with
`python tools/benchmarks/readme_figures.py`; rerun the measurement with
`python tools/benchmarks/tier1_hsx_head_to_head.py`. Full tables, provenance,
and known issues: [docs/performance.rst](docs/performance.rst).

## Parity with SFINCS Fortran v3

Every canonical module was admitted against the reference implementation
(Fortran golden outputs, tiny-grid PETSc matrix dumps, or the retained legacy
implementation) at pinned tolerances that run in CI:

![Measured parity envelopes of the canonical stack](docs/_static/figures/readme/canonical_parity.png)

The scheme-1 monoenergetic `transportMatrix[0,1]` element is pinned to
upstream's expected value because that element is tolerance-unstable in the
Fortran build itself; the `sfincs_jax` direct solve reproduces the expected
value to 4.2e-6 by construction.

## Functionality

| Capability | Status |
| --- | --- |
| RHSMode 1 (fluxes/flows), 2 and 3 (transport matrices) | Supported, Fortran-parity pinned |
| Collisions: pitch-angle scattering, full Fokker-Planck (Rosenbluth) | Supported |
| Trajectory models: full and DKES; radial electric field | Supported |
| Constraint schemes 0 / 1 / 2 | Supported |
| Geometry schemes 1, 2, 3, 4 (three-helicity), 5 (VMEC), 11/12/13 (Boozer, differentiable Fourier) | Supported |
| Solver tiers: structured direct, recycled Krylov (GCROT), host direct referee | Supported (`solve_method="auto"`) |
| Autodiff: `jax.grad`/JVP through geometry, profiles, and the linear solve | Supported (implicit differentiation) |
| `Phi1`/quasineutrality, tangential magnetic drifts | Deferred (served by the retained legacy pipeline) |
| Constraint schemes 3 / 4, mapped speed grids, `export_f` | Deferred (served by the retained legacy pipeline) |
| Non-stellarator-symmetric VMEC | Deferred |

Deferred physics is explicitly out of the canonical stack until each vertical
slice lands with parity evidence; the legacy pipeline
(`sfincs_jax.io.write_sfincs_jax_output_h5` and the full CLI) keeps ownership
of those cases and remains tested. See
[docs/feature_matrix.rst](docs/feature_matrix.rst) for the detailed matrix.

## Optimization showcase

[`examples/optimize_QA_bootstrap.py`](examples/optimize_QA_bootstrap.py) runs a
gradient-based optimization of a quasi-axisymmetric stellarator boundary for
low bootstrap current, where `<j.B>` comes from the kinetic solve: boundary
Fourier coefficients -> `vmec_jax` fixed-boundary equilibrium (implicit-adjoint
VJP) -> differentiable Boozer transform (`booz_xform_jax`) -> `sfincs_jax`
kinetic solve (tier-2 GCROT with implicit differentiation, warm-started and
recycled across optimizer iterations) -> `FSABjHat`. One `jax.value_and_grad`
call differentiates the whole physics chain; the end-to-end gradient is
verified against central finite differences in the example and its CI test
(the kinetic segment agrees to ~3e-6; the full chain to ~1.7e-3, limited by
the host equilibrium solver's termination noise, not by autodiff).

![QA low-bootstrap optimization: objective history, boundary cross-sections, |B| spectrum, and <j.B> profile](docs/_static/figures/readme/optimize_QA_bootstrap.png)

## Examples

Six pedagogic scripts on the canonical API live at the top of
[`examples/`](examples/) — no `main()`, parameters at the top, printed
progress, a plot, and output files written and read back:

- [`run_tokamak.py`](examples/run_tokamak.py) — build a namelist in Python, solve, read HDF5/NetCDF back.
- [`run_w7x.py`](examples/run_w7x.py) — W7-X Boozer geometry with full Fokker-Planck collisions (tier-2 Krylov).
- [`transport_coefficients.py`](examples/transport_coefficients.py) — monoenergetic transport matrices and a collisionality scan.
- [`ambipolar_er_scan.py`](examples/ambipolar_er_scan.py) — scan `Er`, bracket and solve the ambipolar root.
- [`gradients_tour.py`](examples/gradients_tour.py) — `jax.grad` through the solve, verified against finite differences.
- [`optimize_QA_bootstrap.py`](examples/optimize_QA_bootstrap.py) — the flagship optimization above.

The wider `examples/` tree (tutorial notebooks, parity/benchmark drivers,
upstream SFINCS decks) is mapped in [`examples/README.md`](examples/README.md).

## Documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html -W docs docs/_build/html
```

Entry points: [docs/index.rst](docs/index.rst) (landing + quickstart),
[docs/examples.rst](docs/examples.rst),
[docs/performance.rst](docs/performance.rst) (measured evidence and known
issues), [docs/inputs.rst](docs/inputs.rst) / [docs/outputs.rst](docs/outputs.rst)
(namelist and output references), and
[docs/system_equations.rst](docs/system_equations.rst) (the equations solved).

## Known issues

- `Nxi_for_x` ramps embed truncated degrees of freedom as exact zero rows
  (the Fortran code packs them out of its matrix). The differentiable solver
  pins those rows — equivalent to the packed Fortran system; gradients match
  finite differences to 4.4e-8 on the regression deck — and raises at
  execution time if a forward or adjoint solve fails to converge.
- The scheme-1 monoenergetic `transportMatrix[0,1]` element is ill-conditioned
  in the upstream configuration itself (tolerance-unstable in the Fortran
  build); parity for it is pinned to upstream's expected value.

## Testing

```bash
pytest -q
```

## License

See [LICENSE](LICENSE).
