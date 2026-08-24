# 1. Basics — DKX on its own

No optional dependencies, no equilibrium files. Analytic geometry, small grids,
seconds per run. Start with `run_tokamak.py`.

## Running

| script | what it shows |
|---|---|
| `run_tokamak.py` | define a case in Python, solve, read the moments. **Start here.** |
| `run_from_namelist.py` | the same workflow reading an SFINCS deck and writing `sfincsOutput.h5` |
| `inspect_run.py` | print the 115 settable parameters, the moments a run returns, and the plotting entry points |
| `scan_resolution.py` | overriding one parameter on a case — a convergence scan in a loop |

## Plotting

| script | what it draws |
|---|---|
| `plot_summary.py` | `dkx.plot(run)` — the standard figure in one call |
| `plot_custom.py` | your own figure from `run.moments`, in SI units |
| `plot_flux_profile.py` | particle flux, heat flux and bootstrap current against `r/a` |
| `plot_convergence_scan.py` | every resolution axis against the finest grid — the "am I converged?" figure |
| `plot_ambipolar_er.py` | scanning `J_r(E_r)` for the ambipolar root, and the per-species fluxes |
| `plot_monoenergetic.py` | `D11`, `D31`, `D33` against collisionality — the benchmark figure |
| `transport_matrix.py` | the 3×3 matrix, with its Onsager residual as a free error estimate |

Figures land in `output/`.

## Conventions

Every script is module-level top to bottom: no `main()`, no argument parsing.
The knobs are the constants at the top, above the `end of parameters` line, so
running one in Spyder or a notebook means editing a constant and pressing run.
`DKX_CI=1` shrinks the longer scans to a fast smoke test.

Each docstring ends with an **Achieved** line carrying the numbers that script
actually produced, so you can tell a broken install from a working one without
knowing what to expect. Several of those lines exist because the first draft
claimed something the run then contradicted.

Next: [`../2_equilibria`](../2_equilibria) runs the same solver on real
stellarator geometry.
