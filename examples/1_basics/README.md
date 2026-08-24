# 1. Basics — DKX on its own

No optional dependencies, no equilibrium files. Analytic geometry, small grids,
seconds per run. Start with `run_tokamak.py`.

| script | what it shows |
|---|---|
| `run_tokamak.py` | define a case in Python, solve, read the moments. **Start here.** |
| `run_from_namelist.py` | the same workflow reading an SFINCS deck and writing `sfincsOutput.h5` |
| `scan_resolution.py` | overriding one parameter on a case — a convergence scan in a loop |

Every script is module-level top to bottom: no `main()`, no argument parsing.
The knobs are the constants at the top, above the `end of parameters` line.
`DKX_CI=1` shrinks each run to a fast smoke test.

Next: [`../2_equilibria`](../2_equilibria) runs the same solver on real
stellarator geometry.
