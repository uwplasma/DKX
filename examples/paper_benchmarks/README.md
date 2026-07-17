# Paper Benchmarks

Community-standard benchmark cases for the methods paper, in the validation
category: each script produces one publication-style figure plus a JSON record
of the computed coefficients, resolutions, convergence checks, and (when the
Fortran v3 executable is available) direct cross-check numbers.

Scripts are flat and self-documenting: parameters at the top, printed progress,
and a single figure + JSON pair written to
`docs/_static/figures/paper_benchmarks/`.

## Cases

- `monoenergetic_icnts_w7x.py`: ICNTS-style monoenergetic transport
  coefficients (`D11*`, `D31*` versus `nuPrime` at several `EStar`) on the
  W7-X standard configuration at r/a = 0.5, with matched-deck SFINCS Fortran
  v3 cross-check points [C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011)].
- `monoenergetic_icnts_tjii.py`: the same scan on the TJ-II standard
  configuration at s = 0.493 (strong-ripple limit of the benchmark set), with
  the Boozer |B| spectrum supplied through `geometryScheme = 13` and the same
  Fortran v3 cross-check recipe (plus a MUMPS pivot/refinement note recorded
  in the JSON).
- `monoenergetic_icnts_hsx.py`: the same scan on the HSX quasi-helically
  symmetric configuration at r/a = 0.5 (`hsx3free.bc`), showing the
  QH-suppressed 1/nu branch in contrast to W7-X and TJ-II.
- `shaing_callen_convergence.py`: the low-collisionality "hard mode" test --
  the bootstrap coefficient `D31*` on the W7-X standard configuration scanned
  to `nuPrime = 3e-4` at `EStar = 0` and a small finite `EStar`, compared
  against the collisionless Shaing-Callen asymptote evaluated for the same
  surface [K.C. Shaing and J.D. Callen, Phys. Fluids 26, 3315 (1983)], with a
  per-point `Nxi` resolution schedule, split 1.3x convergence gates, and
  Fortran v3 cross-check points.
- `gradient_verification.py`: the AD-vs-FD gradient-verification table
  (three derivatives through the monoenergetic-database, RHSMode=1 solve,
  and ambipolar-root paths; JSON + rst snippet).

## Running

From the repo root:

```bash
python examples/paper_benchmarks/monoenergetic_icnts_w7x.py
```

Set `SFINCS_FORTRAN_EXE=/path/to/sfincs` to enable the Fortran cross-check
points (skipped otherwise).  Equilibrium files are fetched into the local
`sfincs_jax` data cache on first use.  Expect several minutes per script at
production resolution; the CI-sized regression version of each case lives in
the test suite (tests/test_paper_benchmark_monoenergetic in the repo root).
