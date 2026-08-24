## Examples

Start at [`1_basics/`](1_basics) and go down the ladder. The numbering is what
each folder *requires*, not how hard it is — that is objective, where
"beginner/advanced" is a matter of taste.

| Folder | Needs | What's inside |
| --- | --- | --- |
| [`1_basics/`](1_basics) | nothing but `dkx` | define a case in Python, solve, read moments, plot. **Start here.** |
| [`2_equilibria/`](2_equilibria) | real geometry | helical ripple and the 1/ν regime; VMEX → `wout` → DKX in one script |
| [`3_gradients/`](3_gradients) | `jax.grad` | exact derivatives through the solve, checked against finite differences |

Each script is self-contained top to bottom: no `main()`, no argument parsing.
The knobs are constants above the `end of parameters` line, so running one in
Spyder or a notebook means editing a constant and pressing run. Every docstring
ends with an **Achieved** line carrying the numbers that script actually
produced, so a broken install is obvious without knowing what to expect.

### Older topic folders

These predate the ladder above and are being folded into it. They still run.

| Folder | What's inside |
| --- | --- |
| `getting_started/` | CLI and Python output writers, plots, geometry setup |
| `tutorials/` | notebook-led learning path |
| `transport/` | RHSMode=2/3 transport matrices and collisionality scans |
| `autodiff/` | JVP/VJP and implicit differentiation through the solve |
| `optimization/` | neoclassical objectives and candidate screening |
| `vmex_finite_beta/` | finite-beta VMEC, ambipolar `E_r`, Redl, bootstrap current |
| `sfincs_examples/` | vendored upstream SFINCS v3 decks, for parity audits |
| `data/` | small shared input files |

### Not examples

Parity checks, benchmarks and figure generators moved to
[`../tools/`](../tools): they are maintainer scripts that regenerate checked
artifacts, not things to read to learn the code. Look there for
`tools/parity/`, `tools/performance/`, `tools/paper_benchmarks/` and
`tools/publication_figures/`.

### Finding a workflow from the terminal

`workflow_catalog.json` is the machine-readable version of this map, with entry
points, commands, runtime budgets, and whether a workflow needs a local SFINCS
Fortran v3 executable. Tests keep it synchronized with this page.

```bash
python examples/list_workflows.py --list-topics
```

```bash
python examples/list_workflows.py --search "VMEC geometry"
```

Prose documentation lives in `docs/examples.rst`.
