# Contributing

DKX is maintained as scientific software: code, tests, benchmarks and
documentation change together, and every number on these pages names the test,
script or deck that produces it.

## Repository layout

| Path | Holds |
| --- | --- |
| `src/dkx/` | the package ({doc}`how_it_works` maps every module) |
| `tests/` | unit, regression, physics-limit and documentation-contract tests, with frozen SFINCS v3 reference fixtures |
| `examples/` | the numbered example ladder and workflow scripts ({doc}`examples/index`) |
| `tools/` | benchmark, figure and release scripts that regenerate the evidence |
| `docs/` | this documentation (Sphinx, MyST Markdown) |

## Workflow

1. Change one coherent capability.
2. Add or update the tests that pin it, with stable fixtures.
3. Update the page that documents the behaviour, physics scope or workflow.
4. Keep validation reproducible: a new number arrives with its generator.

## Checks before a pull request

```bash
pip install -e ".[dev,docs]"
pytest -q                                           # or the focused test files
PYTHONPATH=src python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Set `PYTHONPATH=src` for the documentation build so autodoc imports the working
tree rather than an installed copy.

The documentation contracts run on every pull request and are quick to run
locally:

```bash
pytest -q tests/test_public_docs_wording_contract.py tests/test_benchmark_doc_claims.py \
  tests/test_readme_budget.py tests/test_readme_quickstart_runs.py tests/test_figure_provenance.py
```

They check that the README and landing-page quickstarts execute, that benchmark
ratios quoted in the text match the checked-in summaries, that the README stays
within its line and word budget, and that the pages read as reference text rather
than as a development log.

## Continuous integration

The `tests` job runs the suite in shards; `coverage` and `coverage-report`
measure line coverage, and the CI fail-under gate is ``80%``. `examples-smoke`
runs the example ladder at its shipped resolution, `wheel-install` installs the
built wheel and source distribution into clean environments and runs a solve
from outside the checkout, and `docs-gate` runs the documentation contracts and
a warning-free Sphinx build.

### Patch coverage on macOS

On macOS `pytest --cov` aborts this suite with `SIGABRT` before pytest prints
anything. Use `coverage run` with the plugin disabled instead:

```bash
pip install -e .
coverage run --source=dkx -m pytest tests/test_your_module.py -q -p no:cov
coverage report --include="*your_module.py" -m
```

A defensive branch that no input can reach is better deleted than excluded.

## Reference fixtures

Some tests compare against PETSc binaries or `sfincsOutput.h5` files produced by
SFINCS v3. To regenerate one, build SFINCS, point `SFINCS_FORTRAN_EXE` at the
executable and run the deck with `saveMatricesAndVectorsInBinary = .true.`.

## Figures

A figure is a claim, and a reader who cannot regenerate it cannot check it.
`docs/figure_provenance.json` maps every displayed figure to the checked-in
script that produces it, and `tests/test_figure_provenance.py` enforces it:

- a displayed figure without a provenance entry fails the suite;
- an entry naming a script that does not exist fails the suite;
- a committed figure that no page displays and no evidence record cites fails
  the suite;
- the number of figures with no identifiable generator is a ratchet: it may
  fall and may not rise.

To add a figure, write the script under `tools/` or `examples/`, add the
provenance entry, then reference the file from the page.
