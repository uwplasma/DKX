# Bring your SFINCS workflow

A SFINCS v3 deck runs unchanged with DKX as the solver. Install with
`pip install -U dkx`; no SFINCS build, MPI or PETSc is needed.

## One run

In a folder holding `input.namelist`:

```console
dkx run
```

This writes `sfincsOutput.h5` in the SFINCS layout, as the SFINCS executable
does, and exits non-zero with a one-line reason if the solve fails. Another
deck or output: `dkx run deck.namelist --out result.h5`.

## Scans from the `!ss` lines

The same `!ss` lines `sfincsScan` reads choose the scan:

```console
dkx scan                                  # scanType 2: Er<value>/sfincsOutput.h5
dkx scan --profiles profiles --out-dir s  # scanType 5: rN_<radius>/Er<value>/
                                          # scanType 1: baseCase, Ntheta21, ...; 3: <variable>_<value>; 4: rN_<radius>
dkx scan --er -2 2 9 --jobs 4 --skip-existing
```

`--radii`, `--er MIN MAX N` and `--profiles` override the deck. Convergence (1:
`<key>MinFactor`, `MaxFactor`, `NumRuns` for `Ntheta`, `Nzeta`, `Nxi`, `NL`, `Nx`,
`NxPotentialsPerVth`, `xMax`, `solverTolerance`) and generic-variable scans (3:
`scanVariable`, `Min`, `Max`, `N`, `Scale`) need no repository checkout. A point
that fails is listed in `scan_failures.json` and the command exits 1; the other
points are kept. From Python:

```python
import dkx

result = dkx.scan("input.namelist", profiles="profiles", er=(-2, 2, 9), jobs=4)
print(result.failures)
```

## The `sfincsScan` scripts

All five scan types (1 convergence, 2 $E_r$, 3 generic variable, 4 radius,
5 radius × $E_r$) and `sfincsScanPlot` run unmodified from
`examples/sfincs_examples/utils` in the DKX repository, with the same `!ss`
lines and `profilesScheme` profile files, against an installed DKX. Run them from
the folder holding the deck, not from inside an old DKX checkout:

```console
python /path/to/DKX/examples/sfincs_examples/utils/sfincsScan --yes
python /path/to/DKX/examples/sfincs_examples/utils/sfincsScanPlot
```

They need no `SFINCS_SYSTEM` and no `job.sfincsScan`; each point is solved
in-process. The scripts are the upstream ones (BSD-style licence, University of Maryland and
the SFINCS development team), ported to Python 3 with a DKX driver in place of the
job submission. Radius scans of type 4 and 5 need `geometryScheme` 5, 11 or 12,
as upstream.

## Folder names

`dkx scan` and `sfincsScan` both name radius folders `rN_0.5`. A scan folder
written by an earlier release as `rN0.5` is reused in place. `sfincsScanPlot` reads both.

Through the `sfincsScan` scripts a failed point does not stop the scan either: it
leaves `dkx_FAILED.txt` with the traceback in the point's folder, the scan prints a
tally of failed points at the end, and a rerun retries only failed or incomplete
points (`tests/test_scan_survives_a_failing_point.py`). `dkx scan` writes the same
information as `scan_failures.json` beside the points.
