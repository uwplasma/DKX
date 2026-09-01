# Examples

Nine numbered rungs, in order. Each is one directory holding a run.py and,
where the same case can be expressed as a file, a case.toml the `dkx` CLI
accepts. Run them top to bottom and every rung changes one thing about the one
before it.

```console
python examples/01_tokamak_profile/run.py
```

| Rung | What it teaches | Runs in |
| --- | --- | --- |
| [`01_tokamak_profile/`](01_tokamak_profile) | the whole native loop: build a `dkx.Case`, `dkx.run`, read SI moments, print the certificate, save NetCDF, plot | ~4 s |
| [`02_vmec_stellarator/`](02_vmec_stellarator) | the same solve on a VMEC `wout`: one field of the case changes | ~3 s |
| [`03_boozer_stellarator/`](03_boozer_stellarator) | Boozer `.bc` geometry, and the solver route the operator's structure picks | ~4 s |
| [`04_monoenergetic_scan/`](04_monoenergetic_scan) | `D11*`, `D31*`, `D33*` against collisionality — the cross-code benchmark figure | ~5 s |
| [`05_ambipolar_profile/`](05_ambipolar_profile) | solving for `E_r` from ambipolarity: every root, classified, with the selection recorded | ~8 s |
| [`06_convergence_certificate/`](06_convergence_certificate) | refining every phase-space axis, and why a small residual is not a converged answer | ~12 s |
| [`07_gradients/`](07_gradients) | `jax.grad` through the solve, checked against central differences | ~17 s |
| [`08_vmex_optimization/`](08_vmex_optimization) | a shape derivative: differentiate the kinetic solve with respect to the `\|B\|` spectrum and descend | ~12 s |
| [`09_phi1_and_impurities/`](09_phi1_and_impurities) | multi-species impurity transport, with and without in-surface potential variation | ~12 s |

## Conventions

Every script is module-level top to bottom: no `main()`, no argument parsing,
no environment switches. The knobs are the constants above the
`end of parameters` line, so running one in an IDE or a notebook means editing
a constant and pressing run. The visible sequence is always the same:

```python
# 1. Imports
# 2. User-editable parameters
# 3. Geometry and species construction
# 4. Physics and numerical configuration
# 5. Run
# 6. Print a scientific summary and certificate
# 7. Save native result
# 8. Plot publication-ready outputs
```

Each rung is sized to run in seconds at the resolution it ships with, so CI
runs the same code a reader does. That resolution buys speed, not accuracy —
rung 06 measures how far from converged the small cases are, and the answer is
"very". Refine before quoting a number.

Where a rung has a case.toml, its run.py builds the same case in Python and
asserts the two have the same deterministic case ID, so the CLI line in its
docstring solves exactly what the script does. Four rungs have no case.toml,
and each says why in its docstring: the monoenergetic workflow, `Phi1`, and
autodiff are reachable from Python but not yet from the native case schema.

Outputs land in `output/<rung>/`, which is gitignored.

## Other trees

| Folder | What's inside |
| --- | --- |
| `sfincs_examples/` | vendored upstream SFINCS v3 decks, kept for parity and benchmark audits, not as a teaching gallery |
| `data/` | small shared input files |
| `getting_started/`, `transport/`, `autodiff/`, `tutorials/`, `optimization/`, `vmex_finite_beta/` | older topic folders that predate the ladder. They still run and other tooling still points at them; the teaching path is the numbered rungs above |

### Not examples

Parity checks, benchmarks and figure generators live in
[`../tools/`](../tools): they regenerate checked artifacts and are maintained
for that, not read to learn the code.

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
